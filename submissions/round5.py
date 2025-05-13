import math
import json
import jsonpickle
import numpy as np
from numpy import random
from collections import deque
from math import log, sqrt, exp
from statistics import NormalDist
from typing import List, Dict, Tuple, Any, Optional
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
normalDist = NormalDist(0, 1)
TRADING_DAYS_PER_YEAR = 365  # 252

class Product:
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    CROISSANTS = "CROISSANTS"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"

VOLCANIC_VOUCHERS_Trade = [
    Product.VOLCANIC_ROCK_VOUCHER_10000,
]

PARAMS = {
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "strike": 10000,
        "mean_volatility": 0.20,
        "total_duration_days": 7,
        "std_window": 30,
        "zscore_threshold": 1.0,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "strike": 10250,
        "mean_volatility": 0.155,
        "total_duration_days": 7,
        "std_window": 30,
        "zscore_threshold": 2.5,
    },
}

class BlackScholes2:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        if volatility <= 0 or time_to_expiry <= 0:
            return max(0, spot - strike)
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * normalDist.cdf(d1) - strike * exp(
            -0 * time_to_expiry
        ) * normalDist.cdf(d2)
        return call_price
    
    @staticmethod
    def implied_volatility(
        target_price,
        spot,
        strike,
        time_to_expiry,
        initial_guess=0.20,
        max_iterations=100,
        tolerance=1e-5,
    ):
        if time_to_expiry <= 0:
            return 0
        volatility = initial_guess
        for i in range(max_iterations):
            try:
                price = BlackScholes2.black_scholes_call(
                    spot, strike, time_to_expiry, volatility
                )
                d1 = (log(spot / strike) + (0.5 * volatility**2) * time_to_expiry) / (
                    volatility * sqrt(time_to_expiry)
                )
                vega = spot * normalDist.pdf(d1) * sqrt(time_to_expiry) / 100
                diff = price - target_price

                if abs(diff) < tolerance:
                    return volatility

                if vega < tolerance:
                    if diff < 0:
                        volatility *= 1.1
                    else:
                        volatility *= 0.9
                    volatility = max(0.0001, min(volatility, 2.0))
                    continue

                vega_unit = spot * normalDist.pdf(d1) * sqrt(time_to_expiry)
                if vega_unit < tolerance:
                    if diff < 0:
                        volatility *= 1.1
                    else:
                        volatility *= 0.9
                    volatility = max(0.0001, min(volatility, 2.0))
                    continue

                volatility = volatility - diff / vega_unit
                if volatility <= 0:
                    volatility = tolerance
            except (ValueError, OverflowError, ZeroDivisionError):
                if "price" in locals() and price < target_price:
                    volatility += 0.01
                else:
                    volatility -= 0.01
                volatility = max(0.0001, min(volatility, 2.0))
        return volatility

class Trader:
    BUY_OD, SELL_OD = dict(), dict()
    POS = dict()
    # total quantity bought or sold in current iteration
    BUYS, SELLS = dict(), dict()
    # fair price of each product
    FAIR_PRICE = dict()
    TIME = None
    # stores the current trader data
    TRADER_DATA = dict()
    # stores the previous trader data
    PREV_TRADER_DATA = dict()
    # communication pipe between different components
    PIPE = dict()
    # safe bid and asks when orderbook empty
    SAFE_EXTREMES = {'B': 0, 'S': 1_000_000}
    MARKET_TRADES = dict()

    def __init__(self, params=None):
        # For Kelp we maintain a history of prices/VWAP values
        self.kelp_prices = []
        self.kelp_vwap = []
        self.last_ema = None  # Store the last EMA value
        self.trader_data = {}

        self.BASKET1_COMPOSITION = {Product.CROISSANTS: 6, Product.JAMS: 3, Product.DJEMBES: 1}
        self.BASKET2_COMPOSITION = {Product.CROISSANTS: 4, Product.JAMS: 2}

        self.basket1_aggression_factor = 1
        self.basket1_ema_span = 150
        self.basket1_ema_alpha = 2 / (self.basket1_ema_span + 1)
        self.basket1_synthetic_ema = None 
        self.basket1_ema_entry_threshold = 140
        self.basket1_ema_exit_threshold = 5.0

        self.basket2_arb_aggression_factor = 1
        self.basket2_ema_span = 150
        self.basket2_ema_alpha = 2 / (self.basket2_ema_span + 1)
        self.basket2_synthetic_ema = None  
        self.basket2_ema_entry_threshold = 90
        self.basket2_ema_exit_threshold = 10  

        self.price_history = {}
        if params is None:
            params = PARAMS
        self.params = params

        self.trader_data = {}

        self.POSITION_LIMITS = {
            Product.KELP: 50,
            Product.SQUID_INK: 50,
            Product.RAINFOREST_RESIN: 50,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.CROISSANTS: 250,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            Product.VOLCANIC_ROCK: 400,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 200,
            Product.MAGNIFICENT_MACARONS: 75,
        }

        self.volcanic_mr_params = {
            "window": 200,  # Lookback window (ticks) for std dev calculation
            "ema_period": 200,  # Desired period (N) for EMA calculation
            "entry_std_dev": 2.7,  # Z-score threshold to enter
            "exit_std_dev": 0.1,  # Z-score threshold to exit (closer to mean)
            "order_size": 5,  # How many units to trade per signal
            "position_limit": 400,  # Max position for THIS strategy component
        }

        if "volcanic_mr" not in self.trader_data:
            self.trader_data["volcanic_mr"] = {
                "price_history": deque(maxlen=self.volcanic_mr_params["window"]),
                "mean": None,  # Keep for reference or potential use
                "std_dev": None,  # Still needed for Z-score volatility
                "current_ema": None,  # To store the calculated EMA
            }

        self.volcanic_mr_params_9500 = {
            "window": 110,  # Lookback window (ticks) for std dev calculation
            "ema_period": 110,  # Desired period (N) for EMA calculation
            "entry_std_dev": 2.1,  # Z-score threshold to enter
            "exit_std_dev": 0.1,  # Z-score threshold to exit (closer to mean)
            "order_size": 5,  # How many units to trade per signal
            "position_limit": 200,  # Max position for THIS strategy component
        }

        if "volcanic_mr_9500" not in self.trader_data:
            self.trader_data["volcanic_mr_9500"] = {
                "price_history": deque(maxlen=self.volcanic_mr_params_9500["window"]),
                "mean": None,  # Keep for reference or potential use
                "std_dev": None,  # Still needed for Z-score volatility
                "current_ema": None,  # To store the calculated EMA
            }

        self.volcanic_mr_params_9750 = {
            "window": 200,
            "ema_period": 200,
            "entry_std_dev": 2.5,
            "exit_std_dev": 0.1,
            "order_size": 5,
            "position_limit": 200, 
        }

        if "volcanic_mr_9750" not in self.trader_data:
            self.trader_data["volcanic_mr_9750"] = {
                "price_history": deque(maxlen=self.volcanic_mr_params_9750["window"]),
                "mean": None,  # Keep for reference or potential use
                "std_dev": None,  # Still needed for Z-score volatility
                "current_ema": None,  # To store the calculated EMA
            }

    def init_vars(self, state: TradingState):
        Trader.TIME = state.timestamp
        # initialize current trader data
        Trader.TRADER_DATA = self.trader_data
        Trader.PIPE = dict()
        if Trader.TIME != 0:
            Trader.PREV_TRADER_DATA = self.trader_data
        else:
            Trader.PREV_TRADER_DATA = self.trader_data
        Trader.MARKET_TRADES = state.market_trades
        # set the total buy and sell amount to empty at start
        Trader.BUYS, Trader.SELLS = dict(), dict()
        # iterate over the products to store the orderbook
        for product in state.order_depths.keys():
            # get the current position for this product and store it
            cpos = state.position.get(product,0)
            Trader.POS[product] = cpos
            # get the orderbook's buy and sell side for this product and store it
            od = state.order_depths[product]
            buy_orders = list(od.buy_orders.items())
            buy_orders = [[p, v] for p, v in od.buy_orders.items()]
            # ascending order of buy prices
            buy_orders.sort(key = lambda x:x[0], reverse = True)
            sell_orders = [[p, v] for p, v in od.sell_orders.items()]
            # ascending order of sell prices
            sell_orders.sort(key = lambda x: x[0]) 
            Trader.BUY_OD[product] = buy_orders
            Trader.SELL_OD[product] = sell_orders
            # compute the fair price for each product 
            Trader.FAIR_PRICE[product] = (
                (Trader.BUY_OD[product][-1][0] if Trader.BUY_OD[product] else Trader.SAFE_EXTREMES['B'])
                + (Trader.SELL_OD[product][-1][0] if Trader.SELL_OD[product] else Trader.SAFE_EXTREMES['S'])
                )/2
        # save the fair prices in the current Trader data
        Trader.TRADER_DATA['FAIR_PRICE'] = Trader.FAIR_PRICE
        
        # initialize the product strategy's useful variables, since we do not have __init__ in product strategy classes
        for product in PRODUCT_STRATEGY: # type: ignore
            for strat in PRODUCT_STRATEGY[product]: # type: ignore
                for var in [
                    'BUY_OD', 'SELL_OD', 'POS', 'BUYS', 'SELLS', 
                    'TIME', 'PREV_TRADER_DATA', 'TRADER_DATA', 'PIPE']:
                    # print(self.FPRICE)
                    setattr(PRODUCT_STRATEGY[product][strat], var, getattr(self, var)) # type: ignore
                    setattr(PRODUCT_STRATEGY[product][strat], 'FPRICE', Trader.FAIR_PRICE.get(product, None)) # type: ignore
                    setattr(PRODUCT_STRATEGY[product][strat], 'PRODUCT', getattr(Product, product)) # type: ignore

    def _get_vwap_and_available_volume(
        self, order_book_side: Dict[int, int], required_volume: int
    ) -> Tuple[Optional[float], int]:
        if not order_book_side or required_volume <= 0:
            return None, 0
        # Work with absolute volumes for calculation simplicity
        abs_volume_side = {p: abs(v) for p, v in order_book_side.items()}
        example_volume = next(iter(order_book_side.values()))
        is_bids = example_volume > 0
        sorted_prices = sorted(
            abs_volume_side.keys(), reverse=is_bids
        )  # bids high to low, asks low to high
        total_volume_filled = 0
        total_cost = 0
        for price in sorted_prices:
            volume_at_level = abs_volume_side[price]
            volume_to_fill = min(volume_at_level, required_volume - total_volume_filled)

            if volume_to_fill <= 0:
                break
            total_volume_filled += volume_to_fill
            total_cost += volume_to_fill * price

            if total_volume_filled >= required_volume:
                break  # Filled the required amount (or more if last level was large)

        if total_volume_filled == 0:
            return None, 0
        # Ensure we didn't overallocate volume due to the last level
        actual_filled_volume = min(total_volume_filled, required_volume)
        achieved_vwap = total_cost / total_volume_filled
        return (
            achieved_vwap,
            actual_filled_volume,
        )  # Return the volume actually available up to required_volume

    def clear_position_order(
        self,
        product: str,
        orders: List[Order],
        state: TradingState,
        buy_order_volume: int,
        sell_order_volume: int,
        fair_value: float,
    ):
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)

        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_quantity = self.POSITION_LIMITS[product] - (position + buy_order_volume)
        sell_quantity = self.POSITION_LIMITS[product] + (position - sell_order_volume)

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

    def RESIN_strat(self, state: TradingState, fair_value: int) -> List[Order]:
        orders: List[Order] = []
        
        position = state.position.get(Product.RAINFOREST_RESIN, 0)
        order_depth = state.order_depths[Product.RAINFOREST_RESIN]
        buy_order_volume = 0
        sell_order_volume = 0

        # Use a list comprehension then fallback if empty.
        sell_prices_above = [
            price for price in order_depth.sell_orders.keys() if price > fair_value + 1
        ]
        if sell_prices_above:
            baaf = min(sell_prices_above)
        else:
            baaf = fair_value + 2

        buy_prices_below = [
            price for price in order_depth.buy_orders.keys() if price < fair_value - 1
        ]
        if buy_prices_below:
            bbbf = max(buy_prices_below)
        else:
            bbbf = fair_value - 2

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, self.POSITION_LIMITS[Product.RAINFOREST_RESIN] - position)  # max amt to buy
                if quantity > 0:
                    orders.append(Order(Product.RAINFOREST_RESIN, int(round(best_ask)), quantity))
                    buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, self.POSITION_LIMITS[Product.RAINFOREST_RESIN] + position)  # should be the max we can sell
                if quantity > 0:
                    orders.append(Order(Product.RAINFOREST_RESIN, int(round(best_bid)), -1 * quantity))
                    sell_order_volume += quantity

        buy_order_volume, sell_order_volume = self.clear_position_order(
            Product.RAINFOREST_RESIN, 
            orders, 
            state, 
            buy_order_volume,
            sell_order_volume,
            fair_value,
        )

        buy_quantity = self.POSITION_LIMITS[Product.RAINFOREST_RESIN] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(Product.RAINFOREST_RESIN, int(round(bbbf + 1)), buy_quantity))

        sell_quantity = self.POSITION_LIMITS[Product.RAINFOREST_RESIN] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(Product.RAINFOREST_RESIN, int(round(baaf - 1)), -sell_quantity))

        return orders

    def KELP_strat(self, state: TradingState, window: int, take_width: int) -> List[Order]:
        orders: List[Order] = []

        position = state.position.get(Product.KELP, 0)
        order_depth = state.order_depths[Product.KELP]
        buy_order_volume = 0
        sell_order_volume = 0

        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price]) >= 17
            ]  # 15
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price]) >= 15
            ]  # 15
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid

            mmmid_price = (mm_ask + mm_bid) / 2
            self.kelp_prices.append(mmmid_price)

            volume = (
                -1 * order_depth.sell_orders[best_ask]
                + order_depth.buy_orders[best_bid]
            )
            vwap = (
                best_bid * (-1) * order_depth.sell_orders[best_ask]
                + best_ask * order_depth.buy_orders[best_bid]
            ) / volume
            self.kelp_vwap.append({"vol": volume, "vwap": vwap})

            if len(self.kelp_vwap) > window:
                self.kelp_vwap.pop(0)

            if len(self.kelp_prices) > window:
                self.kelp_prices.pop(0)

            fair_value = sum([x["vwap"] * x["vol"] for x in self.kelp_vwap]) / sum(
                [x["vol"] for x in self.kelp_vwap]
            )

            fair_value = mmmid_price

            if best_ask <= fair_value - take_width:
                ask_amount = -1 * order_depth.sell_orders[best_ask]
                if ask_amount <= 20:
                    quantity = min(ask_amount, self.POSITION_LIMITS[Product.KELP] - position)
                    if quantity > 0:
                        orders.append(Order(Product.KELP, int(round(best_ask)), quantity))
                        buy_order_volume += quantity
            if best_bid >= fair_value + take_width:
                bid_amount = order_depth.buy_orders[best_bid]
                if bid_amount <= 20:
                    quantity = min(bid_amount, self.POSITION_LIMITS[Product.KELP] + position)
                    if quantity > 0:
                        orders.append(
                            Order(Product.KELP, int(round(best_bid)), -1 * quantity)
                        )
                        sell_order_volume += quantity

            buy_order_volume, sell_order_volume = self.clear_position_order(
                Product.KELP, 
                orders, state,
                buy_order_volume,
                sell_order_volume,
                fair_value
            )

            aaf = [
                price
                for price in order_depth.sell_orders.keys()
                if price > fair_value + 1
            ]
            bbf = [
                price
                for price in order_depth.buy_orders.keys()
                if price < fair_value - 1
            ]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2

            buy_quantity = self.POSITION_LIMITS[Product.KELP] - (position + buy_order_volume)
            if buy_quantity > 0:
                orders.append(Order(Product.KELP, int(round(bbbf + 1)), buy_quantity))  # Buy order

            sell_quantity = self.POSITION_LIMITS[Product.KELP] + (position - sell_order_volume)
            if sell_quantity > 0:
                orders.append(Order(Product.KELP, int(round(baaf - 1)), -sell_quantity))  # Sell order

        return orders

    def calculate_synthetic_details(
        self,
        composition: Dict[str, int],
        order_depths: Dict[str, OrderDepth],
        max_baskets_to_consider: int = 10,
    ) -> Tuple[
        Optional[float], int, Optional[float], int
    ]:  # Standardized default probe volume
        synth_buy_total_vwap_cost = 0
        synth_sell_total_vwap_proceeds = 0
        min_baskets_buyable = float("inf")
        min_baskets_sellable = float("inf")
        data_missing = False

        for product, qty_per_basket in composition.items():
            depth = order_depths.get(product)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                data_missing = True
                break

            # Calculate required volume for probing
            required_volume = qty_per_basket * max_baskets_to_consider
            if required_volume <= 0:
                continue  # Skip if component quantity is zero or negative

            # --- Calculate cost to BUY components (for selling the basket) ---
            comp_buy_vwap, comp_buy_avail_vol = self._get_vwap_and_available_volume(
                depth.sell_orders,  # Hit the asks to buy
                required_volume,
            )
            if comp_buy_vwap is None or comp_buy_avail_vol == 0:
                data_missing = True
                break

            synth_buy_total_vwap_cost += (
                comp_buy_vwap * qty_per_basket
            )  # Cost per basket
            baskets_buyable_this_comp = (
                comp_buy_avail_vol // qty_per_basket
                if qty_per_basket > 0
                else float("inf")
            )
            min_baskets_buyable = min(min_baskets_buyable, baskets_buyable_this_comp)

            # --- Calculate proceeds to SELL components (for buying the basket) ---
            comp_sell_vwap, comp_sell_avail_vol = self._get_vwap_and_available_volume(
                depth.buy_orders,  # Hit the bids to sell
                required_volume,
            )
            if comp_sell_vwap is None or comp_sell_avail_vol == 0:
                data_missing = True
                break

            synth_sell_total_vwap_proceeds += (
                comp_sell_vwap * qty_per_basket
            )  # Proceeds per basket
            baskets_sellable_this_comp = (
                comp_sell_avail_vol // qty_per_basket
                if qty_per_basket > 0
                else float("inf")
            )
            min_baskets_sellable = min(min_baskets_sellable, baskets_sellable_this_comp)

        if (
            data_missing
            or min_baskets_buyable == float("inf")
            or min_baskets_sellable == float("inf")
        ):
            return None, 0, None, 0

        # Ensure we don't report more volume than the initial probe consideration
        final_buy_vol = min(min_baskets_buyable, max_baskets_to_consider)
        final_sell_vol = min(min_baskets_sellable, max_baskets_to_consider)

        # Note: VWAPs are estimates based on probing 'max_baskets_to_consider'. More accuracy might require re-probing with final_buy/sell_vol.
        return (
            synth_sell_total_vwap_proceeds,
            final_sell_vol,
            synth_buy_total_vwap_cost,
            final_buy_vol,
        )

    def _get_mid_price(self, product: Symbol, order_depth: OrderDepth) -> float | None:
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            return (best_ask + best_bid) / 2.0
        return None  # Cannot determine mid-price

    def _get_effective_voucher_price(
        self, voucher_product: Symbol, order_depth: OrderDepth, traderData: Dict
    ) -> float | None:
        mid_price = self._get_mid_price(voucher_product, order_depth)
        if mid_price is not None:
            traderData[voucher_product]["last_price"] = mid_price
            return mid_price
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                if (
                    voucher_product in traderData
                    and "last_price" in traderData[voucher_product]
                ):
                    return traderData[voucher_product]["last_price"]
                return (best_bid + best_ask) / 2.0
            else:
                traderData[voucher_product]["last_price"] = best_bid
                return best_bid
        elif order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            traderData[voucher_product]["last_price"] = best_ask
            return best_ask
        elif (
            voucher_product in traderData
            and "last_price" in traderData[voucher_product]
        ):
            return traderData[voucher_product]["last_price"]
        return None

    def _safe_get_best_bid(self, order_depth: Optional[OrderDepth]) -> Optional[int]:
        return max(order_depth.buy_orders.keys())

    def _safe_get_best_ask(self, order_depth: Optional[OrderDepth]) -> Optional[int]:
        return min(order_depth.sell_orders.keys())

    def _execute_basket_vs_synthetic_spread(
        self,
        basket_symbol: str,
        composition: Dict[str, int],
        target_position: int,
        current_position: int,
        order_depths: Dict[str, OrderDepth],
        aggression: int = 0,
    ) -> Dict[str, List[Order]]:
        orders_by_product: Dict[str, List[Order]] = {basket_symbol: []}
        for product in composition.keys():
            orders_by_product[product] = []

        trade_volume_total = target_position - current_position
        if trade_volume_total == 0:
            return orders_by_product

        is_buy_basket = trade_volume_total > 0
        basket_depth = order_depths.get(basket_symbol)
        if (
            not basket_depth
            or (is_buy_basket and not basket_depth.sell_orders)
            or (not is_buy_basket and not basket_depth.buy_orders)
        ):
            return {}  
        placed_basket_qty_total = 0
        qty_remaining_to_place = abs(trade_volume_total)
        
        layer_percentages = [0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05]

        if is_buy_basket:
            
            sorted_levels = sorted(basket_depth.sell_orders.items())
        else:
            
            sorted_levels = sorted(basket_depth.buy_orders.items(), reverse=True)

        for i, (level_price, level_volume) in enumerate(sorted_levels):
            if qty_remaining_to_place <= 0:
                break

            
            layer_idx = min(
                i, len(layer_percentages) - 1
            )  
            layer_target_qty = max(
                1, int(abs(trade_volume_total) * layer_percentages[layer_idx])
            )

            
            available_at_level = abs(level_volume)

            
            qty_to_place_this_level = min(
                qty_remaining_to_place, layer_target_qty, available_at_level
            )

            if qty_to_place_this_level > 0:
                
                price_adjustment = 0
                order_price = (
                    level_price + price_adjustment
                )  

                order_qty = (
                    qty_to_place_this_level
                    if is_buy_basket
                    else -qty_to_place_this_level
                )
                orders_by_product[basket_symbol].append(
                    Order(basket_symbol, order_price, order_qty)
                )

                placed_basket_qty_total += order_qty  
                qty_remaining_to_place -= qty_to_place_this_level

        if placed_basket_qty_total == 0:
            return {}  

        for product, qty_per_basket in composition.items():
            if product == "DJEMBES" or product == "JAMS" or product == "CROISSANTS":
                continue
            if (
                product == basket_symbol or qty_per_basket == 0
            ):  
                continue
            component_trade_total = -placed_basket_qty_total * qty_per_basket
            if component_trade_total == 0:
                continue

            is_buy_component = component_trade_total > 0
            comp_depth = order_depths.get(product)

            if (
                not comp_depth
                or (is_buy_component and not comp_depth.sell_orders)
                or (not is_buy_component and not comp_depth.buy_orders)
            ):
                continue  
            
            apply_aggression = aggression if product == Product.CROISSANTS else 0

            if is_buy_component:
                best_ask = self._safe_get_best_ask(comp_depth)
                if best_ask is not None:
                    
                    order_price = (
                        best_ask + apply_aggression
                    )  
                    
                    order_qty = int(round(component_trade_total))  
                    if order_qty > 0:
                        orders_by_product[product].append(
                            Order(product, order_price, order_qty)
                        )
            else:  
                best_bid = self._safe_get_best_bid(comp_depth)
                if best_bid is not None:
                    order_price = (
                        best_bid - apply_aggression
                    )  
                    order_qty = int(
                        round(component_trade_total)
                    )  
                    if order_qty < 0:
                        orders_by_product[product].append(
                            Order(product, order_price, order_qty)
                        )
        final_orders = {
            prod: orders for prod, orders in orders_by_product.items() if orders
        }
        return final_orders

    def store_in_traderData(self, name, data):
        strategy_address = self.__class__.__qualname__
        if strategy_address not in self.TRADER_DATA:
            self.TRADER_DATA[strategy_address] = dict()
        self.TRADER_DATA[strategy_address][name] = data

    def get_from_prev_traderData(self, name, product = None, role = None):
        if (product != None and role == None) or (product == None and role != None):
            return None
        if product == None and role == None:
            strategy_address = self.__class__.__qualname__
        else:
            strategy_address = '.'.join([str(product), str(role).capitalize()])
        return self.PREV_TRADER_DATA.get(strategy_address, dict()).get(name, None)
    
    def get_osize(self, state, product = None):
        if not product:
            product = self.PRODUCT
        osize_a = self.POSITION_LIMITS[product] + state.position.get(product,0) - self.SELLS.get(product,0)
        osize_b = self.POSITION_LIMITS[product] - state.position.get(product,0) - self.BUYS.get(product,0)
        return osize_b, osize_a

    def sniper(self, state, fair_price = None, side = None):
        orders = []
        product = self.PRODUCT
        if not fair_price:
            fair_price = self.FPRICE
        buy_orders = self.BUY_OD[product]
        sell_orders = self.SELL_OD[product]
        osize_bid, osize_ask = self.get_osize(state, product)
        
        if side != 'S': 
            for prices, volumes in buy_orders.copy():
                if prices >= fair_price:
                    sell_amt = min(volumes, osize_ask)
                    orders.append(Order(product, prices, -sell_amt))
                    self.SELLS[product] = self.SELLS.get(product, 0) + sell_amt
                    osize_ask -= sell_amt
                    self.POS[product] -= sell_amt
                    if sell_amt == volumes:
                        buy_orders.pop(0)
                    else:
                        buy_orders[0][1] -= sell_amt
                else:
                    break
            self.BUY_OD[product] = buy_orders

        
        if side != 'B': 
            for prices,volumes in sell_orders.copy():
                if prices <= fair_price:
                    buy_amt = min(abs(volumes),osize_bid)
                    orders.append(Order(product, prices, buy_amt ))
                    self.BUYS[product] = self.BUYS.get(product,0) + buy_amt
                    osize_bid -= buy_amt
                    self.POS[product] += buy_amt
                    if buy_amt == abs(volumes):
                        sell_orders.pop(0)
                    else:
                        sell_orders[0][1] += buy_amt
                else:
                    break
            self.SELL_OD[product] = sell_orders
        return orders

    def balancer(self, state, fair_price = None, side = 'None', tol = 0):
        orders = []
        product = self.PRODUCT
        if not fair_price:
            fair_price = self.FPRICE
        buy_orders = self.BUY_OD[product]
        sell_orders = self.SELL_OD[product]
        osize_bid, osize_ask = self.get_osize(state, product)
        
        if side != 'S': 
            for prices,volumes in buy_orders.copy():
                if prices >= fair_price:
                    sell_amt = min(abs(self.POS[product]) - tol,volumes,osize_ask)
                    orders.append(Order(product, prices, -sell_amt))
                    self.SELLS[product] = self.SELLS.get(product,0) + sell_amt
                    osize_ask -= sell_amt
                    self.POS[product] -= sell_amt
                    if sell_amt == volumes:
                        buy_orders.pop(0)
                    else:
                        buy_orders[0][1] -= sell_amt
                else:
                    break
            self.BUY_OD[product] = buy_orders

        if side != 'B': 
            for prices,volumes in sell_orders.copy():
                if prices <= fair_price:
                    buy_amt = min(abs(self.POS[product]) - tol, abs(volumes),osize_bid)
                    orders.append(Order(product, prices, buy_amt ))
                    self.BUYS[product] = self.BUYS.get(product,0) + buy_amt
                    osize_bid -= buy_amt
                    self.POS[product] += buy_amt
                    if buy_amt == abs(volumes):
                        sell_orders.pop(0)
                    else:
                        sell_orders[0][1] += buy_amt
                else:
                    break
            self.SELL_OD[product] = sell_orders
        return orders
    
    def executor(self, state, amount):
        product = self.PRODUCT
        buy_orders = self.BUY_OD[product]
        sell_orders = self.SELL_OD[product]
        orders = []
        osize_bid, osize_ask = self.get_osize(state, product)

        if amount < 0:
            for prices, volumes in buy_orders.copy():
                sell_amt = min(volumes, abs(amount), osize_ask)
                orders.append(Order(product, prices, -sell_amt))
                self.SELLS[product] = self.SELLS.get(product, 0) + sell_amt
                osize_ask -= sell_amt
                amount += sell_amt
                self.POS[product] -= sell_amt
                if sell_amt == volumes:
                    buy_orders.pop(0)
                else:
                    buy_orders[0][1] -= sell_amt
                if amount == 0:
                    break
            self.BUY_OD[product] = buy_orders
            return orders
        if amount > 0:
            for prices, volumes in sell_orders.copy():
                buy_amt = min(abs(volumes), osize_bid, amount)
                orders.append(Order(product, prices, buy_amt))
                self.BUYS[product] = self.BUYS.get(product, 0) + buy_amt
                osize_bid -= buy_amt
                amount -= buy_amt
                self.POS[product] += buy_amt
                if buy_amt == abs(volumes):
                    sell_orders.pop(0)
                else:
                    sell_orders[0][1] += buy_amt
                if amount == 0:
                    break
            self.SELL_OD[product] = sell_orders
            return orders
        return orders
    
    def _handle_basket2_index_arbitrage(
        self,
        basket_symbol: str,
        composition: Dict[str, int],
        current_positions: Dict[str, int],
        order_depths: Dict[str, OrderDepth],
    ) -> Dict[str, List[Order]]:
        orders_to_add: Dict[str, List[Order]] = {}
        basket_pos = current_positions.get(basket_symbol, 0)
        basket_limit = self.POSITION_LIMITS[basket_symbol]
        aggression_factor = getattr(
            self, "basket2_arb_aggression_factor", 1
        )  

        
        probe_volume_baskets = 40  
        synth_bid, _, synth_ask, _ = self.calculate_synthetic_details(
            composition, order_depths, probe_volume_baskets
        )
        synthetic_mid_price = None
        if synth_bid is not None and synth_ask is not None:
            synthetic_mid_price = (synth_bid + synth_ask) / 2.0

        current_ema = self.basket2_synthetic_ema
        if synthetic_mid_price is not None:
            if current_ema is None:  
                current_ema = synthetic_mid_price
            else:  
                current_ema = (
                    self.basket2_ema_alpha * synthetic_mid_price
                    + (1 - self.basket2_ema_alpha) * current_ema
                )
            self.basket2_synthetic_ema = current_ema  

        if current_ema is None:
            return orders_to_add

        basket_depth = order_depths.get(basket_symbol)
        if not basket_depth:
            return orders_to_add

        basket_best_bid = self._safe_get_best_bid(basket_depth)
        basket_best_ask = self._safe_get_best_ask(basket_depth)

        if basket_best_bid is None or basket_best_ask is None:
            return orders_to_add

        basket_entry_ask = basket_best_ask + aggression_factor
        basket_entry_bid = basket_best_bid - aggression_factor

        target_position = None
        exit_triggered = False

        if (
            basket_pos > 0
            and basket_best_bid >= current_ema - self.basket2_ema_exit_threshold
        ):
            target_position = 0  
            exit_triggered = True

        elif (
            basket_pos < 0
            and basket_best_ask <= current_ema + self.basket2_ema_exit_threshold
        ):
            target_position = 0  
            exit_triggered = True

        if not exit_triggered:
            if basket_entry_ask < current_ema - self.basket2_ema_entry_threshold:
                if basket_pos < basket_limit:  # Check position limit
                    target_position = (
                        basket_limit  # Go fully long (adjust sizing logic if needed)
                    )

            # Enter SHORT if basket bid is significantly ABOVE EMA
            elif basket_entry_bid > current_ema + self.basket2_ema_entry_threshold:
                if basket_pos > -basket_limit:  # Check position limit
                    target_position = (
                        -basket_limit
                    )  

        if target_position is not None and target_position != basket_pos:
            basket_orders = self._execute_basket_vs_synthetic_spread(
                basket_symbol,
                composition,
                target_position,
                basket_pos,
                order_depths,
                aggression=aggression_factor,  # Pass aggression
            )
            for product, order_list in basket_orders.items():
                if product not in orders_to_add:
                    orders_to_add[product] = []
                orders_to_add[product].extend(order_list)

        return orders_to_add

    def _handle_basket1_index_arbitrage(
        self,
        basket_symbol: str,
        composition: Dict[str, int],  # Should include DJEMBES for Basket 1
        current_positions: Dict[str, int],
        order_depths: Dict[str, OrderDepth],
    ) -> Dict[str, List[Order]]:
        orders_to_add: Dict[str, List[Order]] = {}
        basket_pos = current_positions.get(basket_symbol, 0)
        basket_limit = self.POSITION_LIMITS[basket_symbol]
        # Use aggression factor defined specifically for Basket 1 EMA strategy
        aggression_factor = getattr(self, "basket1_aggression_factor", 1)

        # 1. Calculate Synthetic Details & Mid-Price
        probe_volume_baskets = 20  # Adjust probe volume if needed for Basket 1
        synth_bid, _, synth_ask, _ = self.calculate_synthetic_details(
            composition, order_depths, probe_volume_baskets
        )
        synthetic_mid_price = None
        if synth_bid is not None and synth_ask is not None:
            synthetic_mid_price = (synth_bid + synth_ask) / 2.0

        # 2. Update Synthetic EMA for Basket 1
        current_ema = self.basket1_synthetic_ema  # Use Basket 1's EMA state
        if synthetic_mid_price is not None:
            if current_ema is None:  # Initialize EMA
                current_ema = synthetic_mid_price
            else:  # Update EMA
                current_ema = (
                    self.basket1_ema_alpha * synthetic_mid_price
                    + (1 - self.basket1_ema_alpha) * current_ema
                )
            self.basket1_synthetic_ema = current_ema  # Store updated EMA for Basket 1

        if current_ema is None:
            return orders_to_add

        basket_depth = order_depths.get(basket_symbol)
        if not basket_depth:
            return orders_to_add

        basket_best_bid = self._safe_get_best_bid(basket_depth)
        basket_best_ask = self._safe_get_best_ask(basket_depth)

        if basket_best_bid is None or basket_best_ask is None:
            return orders_to_add

        basket_entry_ask = basket_best_ask + aggression_factor
        basket_entry_bid = basket_best_bid - aggression_factor

        # 3. EMA-Based Trading Logic
        target_position = None
        exit_triggered = False

        if (
            basket_pos > 0
            and basket_best_bid >= current_ema - self.basket1_ema_exit_threshold
        ):
            target_position = 0  
            exit_triggered = True

        elif (
            basket_pos < 0
            and basket_best_ask <= current_ema + self.basket1_ema_exit_threshold
        ):
            target_position = 0  
            exit_triggered = True

        if not exit_triggered:
            if basket_entry_ask < current_ema - self.basket1_ema_entry_threshold:
                if basket_pos < basket_limit:  
                    target_position = basket_limit

            elif basket_entry_bid > current_ema + self.basket1_ema_entry_threshold:
                if basket_pos > -basket_limit:  
                    target_position = -basket_limit

        if target_position is not None and target_position != basket_pos:
            basket_orders = self._execute_basket_vs_synthetic_spread(
                basket_symbol,
                composition,
                target_position,
                basket_pos,
                order_depths,
                aggression=aggression_factor,
            )
            for product, order_list in basket_orders.items():
                if product not in orders_to_add:
                    orders_to_add[product] = []
                orders_to_add[product].extend(order_list)
        return orders_to_add
    
    def MACARONS_strat(self, state: TradingState):
        orders = []
        conversion_limit = 10

        order_depth = state.order_depths[Product.MAGNIFICENT_MACARONS]
        best_bid = max(order_depth.buy_orders.keys())
        # best_ask = min(order_depth.sell_orders.keys())

        obs = state.observations.conversionObservations[Product.MAGNIFICENT_MACARONS]
        # if obs.bidPrice - obs.transportFees - obs.exportTariff > best_ask: # hardly ever profitable coz of high export tariff
        #     # buy local, sell pristine
        #     buy_vol = min(conversion_limit, -order_depth.sell_orders[best_ask])
        #     orders.append(Order(Product.MAGNIFICENT_MACARONS, best_ask, buy_vol))
        if obs.askPrice + obs.transportFees + obs.importTariff < best_bid:
            sell_vol = min(conversion_limit, order_depth.buy_orders[best_bid])
            orders.append(Order(Product.MAGNIFICENT_MACARONS, best_bid, -sell_vol))
        return orders

    def _trade_generic_mean_reversion(
        self,
        state: TradingState,
        symbol: str,
        params_dict: Dict[str, Any],
        data_dict: Dict[str, Any],
    ) -> List[Order]:
        orders = []
        ema_period = params_dict.get("ema_period", 20)

        order_depth = state.order_depths.get(symbol)
        position = state.position.get(symbol, 0)
        params = params_dict
        data = data_dict
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        mid_price = self._calculate_mid_price(order_depth)
        if mid_price is None:
            return orders

        data["price_history"].append(
            mid_price
        )  

        ema_period = params.get("ema_period", 20)  
        if ema_period <= 0:  
            alpha = 0.1  
        else:
            alpha = 2 / (ema_period + 1)  

        previous_ema = data.get("current_ema")

        if previous_ema is None:
            new_ema = mid_price
        else:
            new_ema = alpha * mid_price + (1 - alpha) * previous_ema

        data["current_ema"] = new_ema  

        if len(data["price_history"]) < params["window"]:
            return orders
        history = np.array(list(data["price_history"]))
        data["std_dev"] = np.std(history)

        if data["std_dev"] < 1e-6:  
            return orders

        z_score = (mid_price - new_ema) / data["std_dev"]

        strat_pos_limit = params_dict.get(
            "position_limit", 0
        )  
        entry_threshold = params["entry_std_dev"]  
        exit_threshold = params["exit_std_dev"]  
        if z_score > entry_threshold and position < strat_pos_limit:
            best_bid = max(order_depth.buy_orders.keys())
            available_qty = order_depth.buy_orders[best_bid]

            max_sell_total = (
                self.POSITION_LIMITS[symbol] + position
            )  

            trade_qty = min(available_qty, max_sell_total)
            if trade_qty > 0:
                orders.append(Order(symbol, best_bid, -trade_qty))

        elif z_score < -entry_threshold and position > -strat_pos_limit:
            best_ask = min(order_depth.sell_orders.keys())
            available_qty = -order_depth.sell_orders[best_ask]
            max_buy_total = (
                self.POSITION_LIMITS[symbol] - position
            )  
            trade_qty = min(available_qty, max_buy_total)
            if trade_qty > 0:
                orders.append(Order(symbol, best_ask, trade_qty))

        elif abs(z_score) < exit_threshold:
            if position > 0:  
                best_bid = max(order_depth.buy_orders.keys())
                available_qty = order_depth.buy_orders[best_bid]
                trade_qty = min(position, available_qty)  
                if trade_qty > 0:
                    orders.append(Order(symbol, best_bid, -trade_qty))
            elif position < 0:  
                best_ask = min(order_depth.sell_orders.keys())
                available_qty = -order_depth.sell_orders[best_ask]
                trade_qty = min(abs(position), available_qty)  
                if trade_qty > 0:
                    orders.append(Order(symbol, best_ask, trade_qty))

        return orders

    def _calculate_mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        """Calculate mid price from order depth, return None if invalid."""
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        try:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        except Exception:
            return None

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        if state.traderData:
            try:
                traderData = jsonpickle.decode(state.traderData)
                self.trader_data = jsonpickle.decode(state.traderData)
            except:
                traderData = {}
        else:
            traderData = {}

        self.init_vars(state)
        for voucher_product in VOLCANIC_VOUCHERS_Trade:
            if voucher_product not in self.trader_data:
                self.trader_data[voucher_product] = {
                    "past_coupon_vol": [],
                    "last_price": None,
                }
            elif "past_coupon_vol" not in self.trader_data[voucher_product]:
                self.trader_data[voucher_product]["past_coupon_vol"] = []

        try:
            result[Product.RAINFOREST_RESIN] = self.RESIN_strat(state, 10000)
        except Exception as e:
            logger.print("Error in RESIN strategy")
            logger.print(e)

        try:
            result[Product.KELP] = self.KELP_strat(state, 10, 1)
        except Exception as e:
            logger.print("Error in KELP strategy")
            logger.print(e)

        try:
            basket1_orders = self._handle_basket1_index_arbitrage(
                Product.PICNIC_BASKET1,
                self.BASKET1_COMPOSITION,
                state.position,
                state.order_depths,
            )
            for product, orders in basket1_orders.items():
                if product not in result:
                    result[product] = []
                result[product].extend(orders)
        except Exception as e:
            logger.print("Error in BASKET 1 strategy")
            logger.print(e)

        try:
            basket2_orders = self._handle_basket2_index_arbitrage(
                Product.PICNIC_BASKET2,
                self.BASKET2_COMPOSITION,
                state.position,
                state.order_depths,
            )
            for product, orders in basket2_orders.items():
                if product not in result:
                    result[product] = []
                result[product].extend(orders)
        except Exception as e:
            logger.print("Error in BASKET 2 strategy")
            logger.print(e)

        voucher_mr_configs = [
            (Product.VOLCANIC_ROCK_VOUCHER_9500, self.volcanic_mr_params_9500, "volcanic_mr_9500"),
            (Product.VOLCANIC_ROCK_VOUCHER_9750, self.volcanic_mr_params_9750, "volcanic_mr_9750"),
            (Product.VOLCANIC_ROCK, self.volcanic_mr_params, "volcanic_mr"),
        ]

        for symbol, params, data_key in voucher_mr_configs:
            if (
                symbol in state.order_depths
                and Product.VOLCANIC_ROCK in state.order_depths
            ):
                voucher_data = self.trader_data.setdefault(
                    data_key, {"price_history": deque(maxlen=params.get("window", 100))}
                )
                mr_orders = self._trade_generic_mean_reversion(
                    state, symbol, params, voucher_data
                )
                if mr_orders:
                    result.setdefault(symbol, []).extend(mr_orders)

        custom_voucher = Product.VOLCANIC_ROCK_VOUCHER_10000
        if (
            custom_voucher in state.order_depths
            and Product.VOLCANIC_ROCK in state.order_depths
        ):
            rock_order_depth = state.order_depths[Product.VOLCANIC_ROCK]
            rock_mid_price = self._calculate_mid_price(rock_order_depth)

            if rock_mid_price is not None:
                voucher_order_depth = state.order_depths[custom_voucher]
                voucher_price = self._get_effective_voucher_price(
                    custom_voucher, voucher_order_depth, self.trader_data
                )
                if voucher_price is not None and voucher_price > 0:
                    tte_years = 3 / 365 - (state.timestamp / 1000000) / 365
                    if tte_years > 1e-6:
                        try:
                            iv = BlackScholes2.implied_volatility(
                                target_price=voucher_price,
                                spot=rock_mid_price,
                                strike=10_000,
                                time_to_expiry=tte_years,
                            )
                        except Exception as e:
                            iv = None
                        if iv is not None:
                            data_key = "vol_10000_zscore"
                            history = self.trader_data.setdefault(
                                data_key, {"history": deque(maxlen=30)}
                            )
                            history["history"].append(iv)
                            iv_list = list(history["history"])
                            if len(iv_list) >= 15:
                                mean_iv = np.mean(iv_list)
                                std_iv = np.std(iv_list)
                                z = (iv - mean_iv) / std_iv if std_iv > 0 else 0
                                position = state.position.get(custom_voucher, 0)
                                limit = self.POSITION_LIMITS[custom_voucher]
                                orders = []

                                if z > 1.0:
                                    best_bid = max(
                                        voucher_order_depth.buy_orders.keys(),
                                        default=None,
                                    )
                                    if best_bid:
                                        qty = min(
                                            voucher_order_depth.buy_orders[best_bid],
                                            limit + position,
                                        )
                                        if qty > 0:
                                            orders.append(
                                                Order(custom_voucher, best_bid, -qty)
                                            )
                                elif z < -1.0:
                                    best_ask = min(
                                        voucher_order_depth.sell_orders.keys(),
                                        default=None,
                                    )
                                    if best_ask:
                                        qty = min(
                                            -voucher_order_depth.sell_orders[best_ask],
                                            limit - position,
                                        )
                                        if qty > 0:
                                            orders.append(
                                                Order(custom_voucher, best_ask, qty)
                                            )

                                if orders:
                                    result.setdefault(custom_voucher, []).extend(orders)

        result[Product.SQUID_INK] = (
            PRODUCT_STRATEGY["SQUID_INK"]["TAKER"].run(state) +  PRODUCT_STRATEGY["SQUID_INK"]["MAKER"].run(state) 
        )
        result[Product.KELP] = (
            PRODUCT_STRATEGY["KELP"]["TAKER"].run(state) +  PRODUCT_STRATEGY["KELP"]["MAKER"].run(state) 
        )
        result[Product.CROISSANTS] = (
            PRODUCT_STRATEGY["CROISSANTS"]["TAKER"].run(state) +  PRODUCT_STRATEGY["CROISSANTS"]["MAKER"].run(state) 
        )
        
        try:
            result[Product.MAGNIFICENT_MACARONS] = self.MACARONS_strat(state)
            current_pos = state.position.get(Product.MAGNIFICENT_MACARONS, 0)
            if current_pos <= 0:
                conversions = min(10, round(-current_pos))
            elif current_pos > 0:
                conversions = max(-10, round(-current_pos))
        except Exception as e:
            logger.print("Error in MACARONS strategy")
            logger.print(e)

        traderData2 = {"kelp_prices": self.kelp_prices, "kelp_vwap": self.kelp_vwap}
        traderData.update(traderData2)
        encoded_trader_data = jsonpickle.encode(self.trader_data)

        logger.flush(state, result, conversions, "")
        return result, conversions, encoded_trader_data


class KELP:
    class Taker(Trader):
        POS_TOL = 25  
        POS_TOL_B = 2  
        OFFLOAD_LIM = 3  
        PRODUCT = "KELP"

        def snipe_and_balance(self, state):
            """
            Takes inefficient orders.
            """

            orders = []
            if self.TIME == 0:
                olivia_bought = False
                olivia_sold = False
            if self.TIME != 0:
                olivia_bought = self.get_from_prev_traderData("olivia_bought")
                olivia_sold = self.get_from_prev_traderData("olivia_sold")

            time = None
            if self.PRODUCT in self.MARKET_TRADES:
                for trade in self.MARKET_TRADES[self.PRODUCT]:
                    if (
                        trade.timestamp == self.TIME - 100
                        or trade.timestamp == self.TIME
                    ):
                        if trade.buyer == "Olivia":
                            olivia_bought = True
                            olivia_sold = False
                            time = trade.timestamp
                        elif trade.seller == "Olivia":
                            olivia_sold = True
                            olivia_bought = False
                            time = trade.timestamp
            self.store_in_traderData("olivia_bought", olivia_bought)
            self.store_in_traderData("olivia_sold", olivia_sold)

            fair_price = self.FPRICE
            self.store_in_traderData("kelp", fair_price)
            if not olivia_bought and not olivia_sold:
                acc_ask = math.ceil(fair_price) + 1
                acc_bid = math.floor(fair_price) - 1
                orders = (
                    orders
                    + self.sniper(state, acc_ask, "B")
                    + self.sniper(state, acc_bid, "S")
                )
            if olivia_bought:
                acc_ask = math.ceil(fair_price) + 1
                acc_bid = math.floor(fair_price)
                orders = (
                    orders
                    + self.sniper(state, acc_ask, "B")
                    + self.sniper(state, acc_bid, "S")
                )
                if self.POS[self.PRODUCT] >= 25:
                    orders = orders + self.balancer(state, self.FPRICE + 1, "B", 25)
            if olivia_sold:
                acc_ask = math.ceil(fair_price)
                acc_bid = math.floor(fair_price) - 1
                orders = (
                    orders
                    + self.sniper(state, acc_ask, "B")
                    + self.sniper(state, acc_bid, "S")
                )
                if self.POS[self.PRODUCT] <= -25:
                    orders = orders + self.balancer(state, self.FPRICE - 1, "S", -25)

            return orders

        def run(self, state: TradingState):
            taker_orders = self.snipe_and_balance(state)

            return taker_orders

    class Maker(Trader):
        LARGE_SIZE = 3  # @HYPERPARAMETER@
        THRESH = 5  # @HYPERPARAMETER@

        def get_market_make_prices(self):
            """
            At what price to market make.
            """
            fair_price = self.FPRICE

            buy_orders = (
                self.BUY_OD[self.PRODUCT]
                if len(self.BUY_OD[self.PRODUCT]) > 0
                else [[int(fair_price) - 3, 20]]
            )
            sell_orders = (
                self.SELL_OD[self.PRODUCT]
                if len(self.SELL_OD[self.PRODUCT]) > 0
                else [[int(fair_price) + 3, 20]]
            )
            acc_bid = math.floor(fair_price) - 3
            acc_ask = math.ceil(fair_price) + 3

            for prices, volumes in sell_orders:
                if prices <= fair_price + 2:
                    if self.POS[self.PRODUCT] < -self.THRESH:
                        acc_ask = prices
                        break
                if prices > fair_price + 1:
                    if (
                        abs(volumes) < self.LARGE_SIZE + 1
                        and self.POS[self.PRODUCT] < -self.THRESH
                    ):
                        acc_ask = prices
                    else:
                        acc_ask = prices - 1
                    break

            for prices, volumes in buy_orders:
                if prices >= fair_price - 2:
                    if self.POS[self.PRODUCT] > self.THRESH:
                        acc_bid = prices
                        break
                if prices < fair_price - 1:
                    if (
                        abs(volumes) < self.LARGE_SIZE + 1
                        and self.POS[self.PRODUCT] > self.THRESH
                    ):
                        acc_bid = prices
                    else:
                        acc_bid = prices + 1
                    break

            acc_ask = max(math.ceil(fair_price), int(acc_ask))
            acc_bid = min(math.floor(fair_price), int(acc_bid))

            return acc_bid, acc_ask

        def market_make(self, state):
            orders = []
            acc_bid, acc_ask = self.get_market_make_prices()
            osize_bid, osize_ask = self.get_osize(state)

            orders.append(Order(self.PRODUCT, acc_ask, -osize_ask))
            orders.append(Order(self.PRODUCT, acc_bid, osize_bid))

            return orders

        def run(self, state: TradingState):
            maker_orders = self.market_make(state)
            return maker_orders


class SQUID_INK:
    class Taker(Trader):
        # write as many helper function and variables as you want here onwards till the start of `run` function...
        POS_TOL = 25  # @HYPERPARAMETER@
        POS_TOL_B = 2  # @HYPERPARAMETER@
        OFFLOAD_LIM = 3  # @HYPERPARAMETER@
        PRODUCT = "SQUID_INK"

        def snipe_and_balance(self, state):
            """
            Takes inefficient orders.
            """
            orders = []
            if self.TIME == 0:
                olivia_bought = False
                olivia_sold = False
            if self.TIME != 0:
                olivia_bought = self.get_from_prev_traderData("olivia_bought")
                olivia_sold = self.get_from_prev_traderData("olivia_sold")

            time = None
            if self.PRODUCT in self.MARKET_TRADES:
                for trade in self.MARKET_TRADES[self.PRODUCT]:
                    if (
                        trade.timestamp == self.TIME - 100
                        or trade.timestamp == self.TIME
                    ):
                        if trade.buyer == "Olivia":
                            olivia_bought = True
                            olivia_sold = False
                            time = trade.timestamp
                        elif trade.seller == "Olivia":
                            olivia_sold = True
                            olivia_bought = False
                            time = trade.timestamp
            self.store_in_traderData("olivia_bought", olivia_bought)
            self.store_in_traderData("olivia_sold", olivia_sold)


            fair_price = self.FPRICE
            self.store_in_traderData("squid_ink", fair_price)
            if olivia_sold:
                acc_ask = math.ceil(fair_price)
                orders = orders + self.sniper(
                    state, acc_ask - 10, "B"
                ) 
            if olivia_bought:
                acc_bid = math.floor(fair_price)
                orders = orders + self.sniper(
                    state, acc_bid + 10, "S"
                )  
            return orders

        def run(self, state: TradingState):
            """
            Runs this taker trading logic for this product at each iteration.
            - Return a list of `Order` objects.
            - Can also return a tuple of `(List[Order], conversions)`.
            """
            # get the taker orders
            taker_orders = self.snipe_and_balance(state)

            # return the taker orders
            return taker_orders

    class Maker(Trader):
        def run(self, state: TradingState):
            return []

class CROISSANTS:
    class Taker(Trader):
        # write as many helper function and variables as you want here onwards till the start of `run` function...
        POS_TOL = 12 # @HYPERPARAMETER@
        POS_TOL_B = 2 # @HYPERPARAMETER@
        OFFLOAD_LIM = 3 # @HYPERPARAMETER@
        PRODUCT = 'CROISSANTS'
        
        def snipe_and_balance(self, state):
            '''
            Takes inefficient orders. 
            '''
            orders = []
            if self.TIME == 0:
                olivia_bought = False
                olivia_sold = False
            if self.TIME!=0:
                olivia_bought = self.get_from_prev_traderData("olivia_bought")
                olivia_sold = self.get_from_prev_traderData("olivia_sold")

            time = None
            if self.PRODUCT in self.MARKET_TRADES:
                for trade in self.MARKET_TRADES[self.PRODUCT]:
                    if trade.timestamp == self.TIME - 100 or trade.timestamp == self.TIME:
                        if trade.buyer == "Olivia":
                            olivia_bought = True
                            olivia_sold = False
                            time = trade.timestamp
                        elif trade.seller == "Olivia":
                            olivia_sold = True
                            olivia_bought = False
                            time = trade.timestamp
            self.store_in_traderData("olivia_bought",olivia_bought)
            self.store_in_traderData("olivia_sold",olivia_sold)
            fair_price = self.FPRICE
            self.store_in_traderData('croissant_price', fair_price)
    
            acc_ask = math.ceil(fair_price) - 10
            acc_bid = math.floor(fair_price) + 10
            if olivia_sold:
                orders = orders + self.sniper(state, acc_ask, 'B') # sell at prices >= acc_ask
            if olivia_bought:
                orders = orders + self.sniper(state, acc_bid, 'S') # buy at prices <= acc_bid
            
            return orders
        
        def run(self, state: TradingState):
            taker_orders = self.snipe_and_balance(state)
            return taker_orders
    
    class Maker(Trader):
        def run(self, state: TradingState):
            return []

PRODUCT_STRATEGY = {
    'KELP': {
            'TAKER': KELP.Taker(),
            'MAKER': KELP.Maker()
        },
    'SQUID_INK': {
            'TAKER': SQUID_INK.Taker(),
            'MAKER': SQUID_INK.Maker()
        },
    'CROISSANTS': {
            'TAKER': CROISSANTS.Taker(),
            'MAKER': CROISSANTS.Maker()
        }
    }
