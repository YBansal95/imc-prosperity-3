import math
import json
import jsonpickle
import numpy as np
from collections import deque
from math import exp
from statistics import NormalDist
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

INF = 1e9
normalDist = NormalDist(0,1)

class Product:
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    CROISSANTS = "CROISSANTS"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"

class Status:
    _position_limit = {
        Product.KELP: 50,
        Product.SQUID_INK: 50,
        Product.RAINFOREST_RESIN: 50,
        Product.JAMS: 350,
        Product.DJEMBES: 60,
        Product.CROISSANTS: 250,
        Product.PICNIC_BASKET1: 60,
        Product.PICNIC_BASKET2: 100,
    }

    _state = None
    _realtime_position = {key:0 for key in _position_limit.keys()}
    
    def __init__(self, product: str) -> None:
        self.product = product

    @classmethod
    def cls_update(cls, state: TradingState) -> None:
        cls._state = state
        for product, posit in state.position.items():
            cls._realtime_position[product] = posit            

    @property
    def position_limit(self) -> int:
        return self._position_limit[self.product]
    @property
    def position(self) -> int:
        return self._state.position.get(self.product, 0)
    @property
    def rt_position(self) -> int:
        return self._realtime_position[self.product]
    
    @classmethod
    def _cls_rt_position_update(cls, product, new_position):
        if abs(new_position) <= cls._position_limit[product]:
            cls._realtime_position[product] = new_position
        else:
            raise ValueError("Position limit exceeded")
    def rt_position_update(self, new_position: int) -> None:
        self._cls_rt_position_update(self.product, new_position)
    
    @property
    def possible_buy_amt(self) -> int:
        return min(self._position_limit[self.product] - self.rt_position, self._position_limit[self.product] - self.position)
    @property
    def possible_sell_amt(self) -> int:
        return min(self._position_limit[self.product] + self.rt_position, self._position_limit[self.product] + self.position)
    @property
    def best_bid(self) -> int:
        return max(self._state.order_depths[self.product].buy_orders.keys()) if self._state.order_depths[self.product].buy_orders else self.best_ask - 1
    @property
    def best_ask(self) -> int:
        return min(self._state.order_depths[self.product].sell_orders.keys()) if self._state.order_depths[self.product].sell_orders else self.best_bid + 1
    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2
    @property
    def total_bidamt(self) -> int:
        """Total amount of all buy orders in the order book"""
        return sum(self._state.order_depths[self.product].buy_orders.values())
    @property
    def total_askamt(self) -> int:
        """Total amount of all sell orders in the order book (positive value)"""
        return -sum(self._state.order_depths[self.product].sell_orders.values())

class LogisticRegressor:
    def __init__(self, window=400, macd_short=12, macd_long=26, macd_signal=9):
        self.window = window

        self.price_window = deque(maxlen=window)
        self.bid_price_window = deque(maxlen=5)
        self.ask_price_window = deque(maxlen=5)
        self.bid_vol_window = deque(maxlen=5)
        self.ask_vol_window = deque(maxlen=5)

        self.macd_short_window = macd_short
        self.macd_long_window = macd_long
        self.macd_signal_window = macd_signal

        
        self.weights = [0.0] * 7
        self.training_data = []

        self.pressure_history = deque(maxlen=window)

    def sigmoid(self, x: float) -> float:
        if x >= 0:
            z = exp(-x)
            return 1 / (1 + z)
        else:
            z = exp(x)
            return z / (1 + z)

    def predict(self, features):
        z = sum(f * w for f, w in zip(features, self.weights))
        return self.sigmoid(z)

    def get_running_pressure(self):
        if len(self.bid_price_window) < 2 or len(self.ask_price_window) < 2:
            return 0.0

        bid_change = self.bid_price_window[-1] - self.bid_price_window[-2]
        ask_change = self.ask_price_window[-1] - self.ask_price_window[-2]

        bid_vol = self.bid_vol_window[-1]
        ask_vol = self.ask_vol_window[-1]

        if bid_change == 0:
            buy_pressure = 0
        elif bid_change > 0:
            buy_pressure = bid_vol
        else:
            buy_pressure = -bid_vol

        if ask_change == 0:
            sell_pressure = 0
        elif ask_change > 0:
            sell_pressure = ask_vol
        else:
            sell_pressure = -ask_vol

        pressure_diff = buy_pressure - sell_pressure
        self.pressure_history.append(pressure_diff)
        if len(self.pressure_history) > self.window:
            self.pressure_history.pop(0)

        return sum(self.pressure_history)

    def calculate_macd(self):
        if len(self.price_window) < self.macd_long_window:
            return 0.0

        prices = np.array(self.price_window)
        short_ema = np.mean(prices[-self.macd_short_window:])
        long_ema = np.mean(prices[-self.macd_long_window:])
        macd_line = short_ema - long_ema

        if len(prices) >= self.macd_long_window + self.macd_signal_window:
            signal_line = np.mean([np.mean(prices[-self.macd_short_window - i:-i]) - np.mean(prices[-self.macd_long_window - i:-i])
                                   for i in range(1, self.macd_signal_window + 1)])
        else:
            signal_line = 0.0

        return macd_line - signal_line

    def extract_features(self) -> list:
        prices = list(self.price_window)
        bids = list(self.bid_price_window)
        asks = list(self.ask_price_window)
        bid_vols = list(self.bid_vol_window)
        ask_vols = list(self.ask_vol_window)

        mid = prices[-1]
        prev_mid = prices[-2] if len(prices) > 1 else mid

        zscore = (mid - np.mean(prices)) / (np.std(prices) + 1e-6)
        price_velocity = mid - prev_mid
        price_acceleration = price_velocity - (prev_mid - prices[-3] if len(prices) > 2 else 0)
        spread = (asks[-1] - bids[-1]) / (asks[-1] + bids[-1] + 1e-6)
        obi = (bid_vols[-1] - ask_vols[-1]) / (bid_vols[-1] + ask_vols[-1] + 1e-6)
        running_pressure = self.get_running_pressure()
        macd = self.calculate_macd()

        return [zscore, price_velocity, price_acceleration, spread, obi, running_pressure, macd]

    def update(self, mm_mid, bid_price, ask_price, bid_vol, ask_vol):
        mid_price = mm_mid

        self.bid_price_window.append(bid_price)
        self.ask_price_window.append(ask_price)
        self.bid_vol_window.append(bid_vol)
        self.ask_vol_window.append(ask_vol)
        self.price_window.append(mid_price)

        if len(self.price_window) > self.window:
            self.price_window.pop(0)

        if len(self.bid_price_window) > 5:
            self.bid_price_window.pop(0)
            self.bid_vol_window.pop(0)
            self.ask_price_window.pop(0)
            self.ask_vol_window.pop(0)

        if len(self.price_window) < self.window: 
            return "hold"

        features = self.extract_features()
        prob = self.predict(features)

        action = "hold"
        
        volatility = np.std(self.price_window)
        buy_threshold = 0.5 + min(volatility / 50, 0.1)
        sell_threshold = 0.5 - min(volatility / 50, 0.1)

        if prob > buy_threshold:
            action = "buy"
        elif prob < sell_threshold:
            action = "sell"
        return action

    def add_training_example(self):
        if len(self.price_window) < self.window: 
            return
        
        features = self.extract_features()
        prices = list(self.price_window)
        label = int(prices[-1] > prices[-2])

        if len(self.training_data) > self.window:
            self.training_data = self.training_data[-self.window:]
        self.training_data.append((features, label))

    def train_model(self, lr=0.01, epochs=10):
        if len(self.training_data) < 5:
            return

        X = np.array([x[0] for x in self.training_data])
        y = np.array([x[1] for x in self.training_data])
        weights = np.array(self.weights)

        for _ in range(epochs):
            for xi, target in zip(X, y):
                pred = self.sigmoid(np.dot(xi, weights))
                gradient = (pred - target) * xi
                weights -= lr * gradient

        self.weights = weights.tolist()

class Trader:
    def __init__(self):
        self.kelp_prices = []
        self.kelp_vwap = []
        
        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50, 
            Product.KELP: 50,
            Product.SQUID_INK: 50,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
        }

        self.squid_params = {
            "lr_window": 200,
            "soft_pos_limit": int(0.9 * self.LIMIT[Product.SQUID_INK]),
            "macd_short": 12,
            "macd_long": 26,
            "macd_signal": 9,
            "rate": 0.01,
            "epochs": 10,
        }
        
        self.squid_avg_entry = None
        self.squid_pnl = 0
        self.squid_ink_strat = LogisticRegressor(
            self.squid_params["lr_window"],
            self.squid_params["macd_short"],
            self.squid_params["macd_long"],
            self.squid_params["macd_signal"],
        )

        self.state_croissant = Status(Product.CROISSANTS)
        self.state_jam = Status(Product.JAMS)
        self.state_djembe = Status(Product.DJEMBES)
        self.state_basket1 = Status(Product.PICNIC_BASKET1)
        self.state_basket2 = Status(Product.PICNIC_BASKET2)

    def get_mm_mid(self, state: TradingState, product, adverse_vol=15):
        b_orders = state.order_depths[product].buy_orders
        s_orders = state.order_depths[product].sell_orders

        if not b_orders or not s_orders:
            return None

        best_bid = max(b_orders.keys())
        best_ask = min(s_orders.keys())

        top_bids = [p for p in b_orders if b_orders[p] >= adverse_vol]
        top_asks = [p for p in s_orders if abs(s_orders[p]) >= adverse_vol]

        mm_bid = max(top_bids) if top_bids else best_bid
        mm_ask = min(top_asks) if top_asks else best_ask

        return (mm_bid + mm_ask) / 2

    def SQUID_strat(self, state: TradingState):
        orders = []

        orderdepth = state.order_depths[Product.SQUID_INK]
        best_bid = max(orderdepth.buy_orders.keys()) if orderdepth.buy_orders else None
        best_ask = min(orderdepth.sell_orders.keys()) if orderdepth.sell_orders else None

        mid = self.get_mm_mid(state, Product.SQUID_INK, adverse_vol=20) 
        action = self.squid_ink_strat.update(mid, best_bid, best_ask, orderdepth.buy_orders[best_bid], orderdepth.sell_orders[best_ask])
        self.squid_ink_strat.add_training_example()

        if state.timestamp % 5000 == 0:
            self.squid_ink_strat.train_model(self.squid_params["rate"], self.squid_params["epochs"])

        if len(self.squid_ink_strat.price_window) >= 2:
            fair_value = (self.squid_ink_strat.price_window[-1] + self.squid_ink_strat.price_window[-2]) / 2
        else:
            fair_value = mid  

        trades = state.own_trades.get(Product.SQUID_INK, [])
        for trade in trades:
            if trade.timestamp != state.timestamp - 100:
                continue
            if trade.buyer == 'SUBMISSION':
                self.squid_pnl -= trade.price * trade.quantity
            else:
                self.squid_pnl += trade.price * trade.quantity

        pos = state.position.get(Product.SQUID_INK, 0)
        if pos == 0:
            self.squid_pnl = 0
            self.squid_avg_entry = None
        else:
            self.squid_avg_entry = abs(self.squid_pnl / pos)

        pos = state.position.get(Product.SQUID_INK, 0)
        b_qty = min(20, self.squid_params["soft_pos_limit"] - pos)
        s_qty = max(-20, -self.squid_params["soft_pos_limit"] - pos)
        b_price = min(int(fair_value), best_bid + 1, best_ask - 2)
        s_price = max(int(fair_value), best_ask - 1, best_bid + 2)

        if action.startswith('buy'):
            orders.append(Order(Product.SQUID_INK, b_price, b_qty))

        elif action.startswith('sell'):
            orders.append(Order(Product.SQUID_INK, s_price, s_qty))

        return orders

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
            buy_order_volume, sell_order_volume, fair_value
        )

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            
            orders.append(Order(Product.RAINFOREST_RESIN, int(round(bbbf + 1)), buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            
            orders.append(Order(Product.RAINFOREST_RESIN, int(round(baaf - 1)), -sell_quantity))

        return orders

    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int, position_limit: int, product: str, buy_order_volume: int, sell_order_volume: int, fair_value: float) -> List[Order]:
        
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
    
    def kelp_orders(self, order_depth: OrderDepth, timespan:int, kelp_take_width: float, position: int, position_limit: int) -> List[Order]:
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
        
            fair_value = sum([x["vwap"]*x['vol'] for x in self.kelp_vwap]) / sum([x['vol'] for x in self.kelp_vwap])
            
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

            buy_order_volume, sell_order_volume = self.clear_position_order(orders, order_depth, position, position_limit, Product.KELP, buy_order_volume, sell_order_volume, fair_value)
            
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

    def trade_jams(self, state: TradingState) -> List[Order]:
        product = Product.JAMS
        orders = []

        if product not in state.order_depths:
            return orders

        depth = state.order_depths[product]
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        position = state.position.get(product, 0)
        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid

        COMPONENT_BUY_ZSCORE = -2.5
        COMPONENT_SELL_ZSCORE = 1.0
        COMPONENT_SHORT_BIAS = True
        POSITION_LIMIT = self.LIMIT[Product.JAMS]
        POSITION_RISK_FACTOR = 0.7
        BASE_TRADE_SIZE = 15

        if spread < 2:
            return orders

        if not hasattr(self, "jams_price_history"):
            self.jams_price_history = []
        self.jams_price_history.append(mid_price)
        if len(self.jams_price_history) > 100:
            self.jams_price_history.pop(0)

        if len(self.jams_price_history) < 10:
            return orders

        long_capacity = POSITION_LIMIT - position
        short_capacity = POSITION_LIMIT + position
        if long_capacity <= 0 and short_capacity <= 0:
            return orders

        recent_prices = self.jams_price_history[-30:]
        price_mean = np.mean(recent_prices)
        price_std = np.std(recent_prices) if len(recent_prices) > 5 else 1
        z_score = (mid_price - price_mean) / max(price_std, 0.1)

        recent_avg = np.mean(self.jams_price_history[-5:])
        older_avg = np.mean(self.jams_price_history[-15:-5])
        trend = recent_avg - older_avg

        position_ratio = abs(position) / POSITION_LIMIT
        risk_adjustment = max(0.3, 1 - position_ratio * POSITION_RISK_FACTOR)

        product_orders = []

        if z_score < COMPONENT_BUY_ZSCORE and long_capacity > 0 and (trend > 0 or not COMPONENT_SHORT_BIAS):
            intensity = min(2, max(1, abs(z_score) / 2))
            trade_size = int(BASE_TRADE_SIZE * intensity * risk_adjustment)
            trade_size = min(trade_size, long_capacity)
            trade_size = max(1, trade_size)
            product_orders.append(Order(product, best_ask, trade_size))
            if hasattr(self, "trade_counter"):
                self.trade_counter[product] += 1

        elif (z_score > COMPONENT_SELL_ZSCORE or trend < -0.1) and short_capacity > 0:
            intensity = min(3, max(1, abs(z_score) + (1 if trend < 0 else 0)))
            trade_size = int(BASE_TRADE_SIZE * intensity * risk_adjustment)
            trade_size = min(trade_size, short_capacity)
            trade_size = max(1, trade_size)
            product_orders.append(Order(product, best_bid, -trade_size))
            if hasattr(self, "trade_counter"):
                self.trade_counter[product] += 1

        if product_orders:
            orders.extend(product_orders)

        return orders
    
    def index_arb(self, basket: Status, components: list, weights: list, threshold: float):
        synthetic = sum(w * c.mid for w, c in zip(weights, components))
        spread = basket.mid - synthetic
        orders = []
        if spread > threshold:
            sell_amt = min(basket.possible_sell_amt, basket.total_bidamt)
            if sell_amt > 0:
                orders.append(Order(basket.product, basket.best_bid, -sell_amt))
        elif spread < -threshold:
            buy_amt = min(basket.possible_buy_amt, basket.total_askamt)
            if buy_amt > 0:
                orders.append(Order(basket.product, basket.best_ask, buy_amt))
        return orders

    def run(self, state: TradingState):
        result = {}
        Status.cls_update(state)
            
        if Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            resin_orders_list = self.resin_orders(
                state.order_depths[Product.RAINFOREST_RESIN], 10000,
                resin_position, self.LIMIT[Product.RAINFOREST_RESIN]
            )
            result[Product.RAINFOREST_RESIN] = resin_orders_list

        if Product.KELP in state.order_depths:
            kelp_position = state.position.get(Product.KELP, 0)
            kelp_orders_list = self.kelp_orders(state.order_depths[Product.KELP], 10, 1, kelp_position, self.LIMIT[Product.KELP])
            result[Product.KELP] = kelp_orders_list
        
        if Product.SQUID_INK in state.order_depths:
            result[Product.SQUID_INK] = self.SQUID_strat(state)

        if Product.JAMS in state.order_depths:
            result[Product.JAMS] = self.trade_jams(state)

        components_b1 = [self.state_croissant, self.state_jam, self.state_djembe]
        weights_b1 = [6, 3, 1]
        if Product.PICNIC_BASKET1 in state.order_depths:
            basket1_orders = self.index_arb(self.state_basket1, components_b1, weights_b1, 48)#50,48
            result[Product.PICNIC_BASKET1] = basket1_orders
        
        # components_b2 = [self.state_croissant, self.state_jam]
        # weights_b2 = [4, 2]
        # if "PICNIC_BASKET2" in state.order_depths:
        #     basket2_orders = Strategy.index_arb(self.state_basket2, components_b2, weights_b2, 59)#30,59
        #     result["PICNIC_BASKET2"] = basket2_orders

        traderData = jsonpickle.encode({
            "kelp_prices": self.kelp_prices,
            "kelp_vwap": self.kelp_vwap,
        })

        conversions = 0
        logger.flush(state, result, conversions, "")
        return result, conversions, traderData


