import math
import json
import jsonpickle
from typing import List, Any
from datamodel import OrderDepth, TradingState, Order, Symbol, Listing, Trade, Observation, ProsperityEncoder

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out

logger = Logger()

class Product:
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"

class Trader:
    def __init__(self):
        self.kelp_prices = []
        self.kelp_vwap = []
        
        self.traderData = {}
        self.squid_mid_history = []

        self.POSITION_LIMITS = {
            Product.KELP: 50,
            Product.SQUID_INK: 50,
            Product.RAINFOREST_RESIN: 50,
        }

    def resin_orders(self, order_depth: OrderDepth, fair_value: int, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0
        
        sell_prices_above = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        if sell_prices_above:
            baaf = min(sell_prices_above)
        else:
            baaf = fair_value + 2

        buy_prices_below = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        if buy_prices_below:
            bbbf = max(buy_prices_below)
        else:
            bbbf = fair_value - 2

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit - position)  
                if quantity > 0:
                    orders.append(Order(Product.RAINFOREST_RESIN, int(round(best_ask)), quantity))
                    buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, position_limit + position)  
                if quantity > 0:
                    orders.append(Order(Product.RAINFOREST_RESIN, int(round(best_bid)), -1 * quantity))
                    sell_order_volume += quantity
        
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, Product.RAINFOREST_RESIN,
            buy_order_volume, sell_order_volume, fair_value, 1)

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(Product.RAINFOREST_RESIN, int(round(bbbf + 1)), buy_quantity))  

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(Product.RAINFOREST_RESIN, int(round(baaf - 1)), -sell_quantity))  

        return orders
    
    def clear_position_order(
        self, orders: List[Order], order_depth: OrderDepth, position: int, position_limit: int,
        product: str, buy_order_volume: int, sell_order_volume: int, fair_value: float, width: int
    ) -> List[Order]:
        
        position_after_take = position + buy_order_volume - sell_order_volume
        fair = round(fair_value)
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, int(fair_for_ask), -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, int(fair_for_bid), abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
    
        return buy_order_volume, sell_order_volume

    def kelp_orders(self, order_depth: OrderDepth, timespan: int, kelp_take_width: float, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0

        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:    
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 21]  
            filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 15]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid
            
            mmmid_price = (mm_ask + mm_bid) / 2    
            self.kelp_prices.append(mmmid_price)

            volume = -1 * order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            vwap = (best_bid * (-1) * order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[best_bid]) / volume
            self.kelp_vwap.append({"vol": volume, "vwap": vwap})
            
            if len(self.kelp_vwap) > timespan:
                self.kelp_vwap.pop(0)
            
            if len(self.kelp_prices) > timespan:
                self.kelp_prices.pop(0)
        
            fair_value = mmmid_price

            if best_ask <= fair_value - kelp_take_width:
                ask_amount = -1 * order_depth.sell_orders[best_ask]
                if ask_amount <= 20:
                    quantity = min(ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(Product.KELP, int(round(best_ask)), quantity))
                        buy_order_volume += quantity
            if best_bid >= fair_value + kelp_take_width:
                bid_amount = order_depth.buy_orders[best_bid]
                if bid_amount <= 20:
                    quantity = min(bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(Product.KELP, int(round(best_bid)), -1 * quantity))
                        sell_order_volume += quantity

            buy_order_volume, sell_order_volume = self.clear_position_order(
                orders, order_depth, position, position_limit, Product.KELP,
                buy_order_volume, sell_order_volume, fair_value, 2)
            
            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2
           
            buy_quantity = position_limit - (position + buy_order_volume)
            if buy_quantity > 0:
                orders.append(Order(Product.KELP, int(round(bbbf + 1)), buy_quantity))  

            sell_quantity = position_limit + (position - sell_order_volume)
            if sell_quantity > 0:
                orders.append(Order(Product.KELP, int(round(baaf - 1)), -sell_quantity))  

        return orders

    def get_running_pressure(self, order_depth: OrderDepth, trader_data: dict) -> float:
        if "last_best_bid" not in trader_data:
            trader_data["last_best_bid"] = None
        if "last_best_ask" not in trader_data:
            trader_data["last_best_ask"] = None
        if "last_best_bid_volume" not in trader_data:
            trader_data["last_best_bid_volume"] = 0
        if "last_best_ask_volume" not in trader_data:
            trader_data["last_best_ask_volume"] = 0
        if "pressure_history" not in trader_data:
            trader_data["pressure_history"] = []

        best_bid = None
        best_ask = None
        best_bid_volume = 0
        best_ask_volume = 0

        for bid, volume in order_depth.buy_orders.items():
            if volume >= 15:
                if best_bid is None or bid > best_bid:
                    best_bid = bid
                    best_bid_volume = volume

        for ask, volume in order_depth.sell_orders.items():
            if abs(volume) >= 15:
                if best_ask is None or ask < best_ask:
                    best_ask = ask
                    best_ask_volume = abs(volume)

        if best_bid is None or best_ask is None:
            return 0.0

        buy_pressure = 0
        sell_pressure = 0

        if best_bid is not None and trader_data["last_best_bid"] is not None:
            if best_bid > trader_data["last_best_bid"]:
                buy_pressure = best_bid_volume
            elif best_bid == trader_data["last_best_bid"]:
                buy_pressure = best_bid_volume - trader_data["last_best_bid_volume"]
            else:
                buy_pressure = -trader_data["last_best_bid_volume"]

        if best_ask is not None and trader_data["last_best_ask"] is not None:
            if best_ask < trader_data["last_best_ask"]:
                sell_pressure = best_ask_volume
            elif best_ask == trader_data["last_best_ask"]:
                sell_pressure = best_ask_volume - trader_data["last_best_ask_volume"]
            else:
                sell_pressure = -trader_data["last_best_ask_volume"]

        pressure_difference = buy_pressure - sell_pressure
        trader_data["pressure_history"].append(pressure_difference)
        if len(trader_data["pressure_history"]) > 50:
            trader_data["pressure_history"].pop(0)

        running_pressure = sum(trader_data["pressure_history"]) if len(trader_data["pressure_history"]) == 50 else 0

        trader_data["last_best_bid"] = best_bid
        trader_data["last_best_ask"] = best_ask
        trader_data["last_best_bid_volume"] = best_bid_volume
        trader_data["last_best_ask_volume"] = best_ask_volume

        return running_pressure
    
    def squid_ink_orders(self, position: int, timestamp: int, signal: str, fair_value: float) -> List[Order]:
        if Product.SQUID_INK not in self.traderData:
            self.traderData[Product.SQUID_INK] = {
                "upper_edge": 3,   
                "lower_edge": 3,    
                "volume_history": [],  
                "last_position": 0,   
                "optimized": False,    
                "last_signal": None     
            }
        
        available_volume = self.POSITION_LIMITS[Product.SQUID_INK] - abs(position)
        if available_volume <= 0:
            return []  
        
        orders = []
        state_data = self.traderData[Product.SQUID_INK]
        filled_volume = abs(position - state_data["last_position"])
        state_data["last_position"] = position

        if timestamp > 0:
            state_data["volume_history"].append(filled_volume)
            if len(state_data["volume_history"]) > 5:
                state_data["volume_history"].pop(0)
            
            if state_data["last_signal"] != signal:
                state_data["optimized"] = False
                state_data["volume_history"] = []
            
            if len(state_data["volume_history"]) >= 3 and not state_data["optimized"]:
                volume_avg = sum(state_data["volume_history"]) / len(state_data["volume_history"])
                volume_bar = 1  
                
                if signal == "b":
                    curr_edge = state_data["lower_edge"]
                    if volume_avg > volume_bar:
                        state_data["lower_edge"] = min(curr_edge + 1, 5)
                        state_data["volume_history"] = []
                    elif volume_avg < volume_bar * 0.7:
                        state_data["lower_edge"] = max(curr_edge - 1, -5)
                        state_data["volume_history"] = []
                else: 
                    curr_edge = state_data["upper_edge"]
                    if volume_avg > volume_bar:
                        state_data["upper_edge"] = min(curr_edge + 1, 5)
                        state_data["volume_history"] = []
                    elif volume_avg < volume_bar * 0.7:
                        state_data["upper_edge"] = max(curr_edge - 1, -5)
                        state_data["volume_history"] = []
        
        state_data["last_signal"] = signal
        upper_edge = state_data["upper_edge"]
        lower_edge = state_data["lower_edge"]
        
        if signal == "b":
            target_volume = self.POSITION_LIMITS[Product.SQUID_INK] - position
            if target_volume > 0:
                return [Order(Product.SQUID_INK, fair_value - lower_edge, int(round(target_volume)))]
            else:
                return []
        elif signal == "s":
            target_volume = -self.POSITION_LIMITS[Product.SQUID_INK] - position
            if target_volume < 0:
                return [Order(Product.SQUID_INK, fair_value + upper_edge, int(round(target_volume)))]
            else:
                return []
        else:
            return []

    def squid_orders(self, state: TradingState) -> List[Order]:
        squid_position = state.position.get(Product.SQUID_INK, 0)
        order_depth = state.order_depths[Product.SQUID_INK]

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 20]
        filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 20]
        mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask
        mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid
        mmmid_price = (mm_ask + mm_bid) / 2
        self.squid_mid_history.append(mmmid_price)
        self.squid_mid_history = self.squid_mid_history[-4:]
        if len(self.squid_mid_history) >= 2:
            fair_value = round((self.squid_mid_history[-1] + self.squid_mid_history[-2]) / 2)
        else:
            fair_value = mmmid_price

        running_pressure = self.get_running_pressure(order_depth, self.traderData)
        if (running_pressure > 0 and running_pressure < 30) or (running_pressure < -30):
            signal = "b"
        elif (running_pressure < 0 and running_pressure > -30) or (running_pressure > 30):
            signal = "s"
        else:
            signal = "h"

        return self.squid_ink_orders(squid_position, state.timestamp, signal, fair_value)
    
    def run(self, state: TradingState):
        result = {}

        if Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            resin_orders_list = self.resin_orders(
                state.order_depths[Product.RAINFOREST_RESIN], 10000,
                resin_position, self.POSITION_LIMITS[Product.RAINFOREST_RESIN]
            )
            result[Product.RAINFOREST_RESIN] = resin_orders_list

        if Product.KELP in state.order_depths:
            kelp_position = state.position.get(Product.KELP, 0)
            kelp_orders_list = self.kelp_orders(state.order_depths[Product.KELP], 10, 1, kelp_position, self.POSITION_LIMITS[Product.KELP])
            result[Product.KELP] = kelp_orders_list

        if Product.SQUID_INK in state.order_depths:
            result[Product.SQUID_INK] = self.squid_orders(state)

        traderData = jsonpickle.encode({
            "kelp_prices": self.kelp_prices, 
            "kelp_vwap": self.kelp_vwap,
            "squid_mid_history": self.squid_mid_history,
            "traderData": self.traderData
        })

        conversions = 0
        logger.flush(state, result, conversions, "")
        return result, conversions, traderData
