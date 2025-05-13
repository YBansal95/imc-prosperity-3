import math
import json
import math
import jsonpickle
import numpy as np
from collections import deque
from abc import abstractmethod
from math import log, sqrt, exp
from statistics import NormalDist
from typing import List, Any, Dict, TypeAlias
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
JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

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

VOLCANIC_VOUCHERS_Trade = [
    Product.VOLCANIC_ROCK_VOUCHER_10000,
    Product.VOLCANIC_ROCK_VOUCHER_10250,
]

PARAMS = {
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "strike": 10000,
        "mean_volatility": 0.20,
        "total_duration_days": 7,
        "std_window": 30,
        "zscore_threshold": 1.0
    },

    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "strike": 10250,
        "mean_volatility": 0.20,
        "total_duration_days": 7,
        "std_window": 30,
        "zscore_threshold": 2.0
    },
    
    Product.SQUID_INK: {
        "window_size": 63,
        "ema_alpha": 0.12,
        "z_score_threshold": 1.1,
        "trade_size": 6,
    },
}

class PicnicBasketStrategy:
    def __init__(self):
        self.POSITION_LIMITS = {
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.CROISSANTS: 250,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100
        }

        self.BASKET1_COMPOSITION = {Product.CROISSANTS: 6, Product.JAMS: 3, Product.DJEMBES: 1}
        self.BASKET2_COMPOSITION = {Product.CROISSANTS: 4, Product.JAMS: 2}
        self.MAX_ORDER_SIZES = {
            Product.JAMS: 30,            
            Product.DJEMBES: 3,          
            Product.CROISSANTS: 20,
            Product.PICNIC_BASKET1: 2,   
            Product.PICNIC_BASKET2: 4   
        }

        self.price_history = {product: [] for product in self.POSITION_LIMITS.keys()}
        self.basket_spread_history = []
        
        self.entry_info = {
            Product.PICNIC_BASKET2: None  
        }

        self.fair_values = {}
        self.last_midprices = {}
        self.price_trends = {product: 0 for product in self.POSITION_LIMITS.keys()}

        self.BASKET1_BUY_THRESHOLD = -0.01   
        self.BASKET1_SELL_THRESHOLD = 0.008  
        self.BASKET2_BUY_THRESHOLD = -0.02   
        self.BASKET2_SELL_THRESHOLD = 0.008  

        self.BASKET2_MAX_LONG_POSITION = 10   
        self.BASKET2_MAX_SHORT_POSITION = 30  
        
        self.COMPONENT_PARAMS = {
            Product.CROISSANTS: {
                "BUY_ZSCORE": -3.8,      
                "SELL_ZSCORE": 2.8,        
                "TREND_SENSITIVITY": 0.8, 
                "POSITION_RISK_FACTOR": 0.9,  
                "MAX_TRADE_SIZE": 10,      
                "MIN_SPREAD": 1.0          
            },
            Product.JAMS: {
                "BUY_ZSCORE": -4.8,        
                "SELL_ZSCORE": 4.0,        
                "TREND_SENSITIVITY": 0.15,  
                "POSITION_RISK_FACTOR": 0.5,  
                "MAX_TRADE_SIZE": 25,      
                "MIN_SPREAD": 2.0          
            },
            Product.DJEMBES: {
                "BUY_ZSCORE": -0.5,        
                "SELL_ZSCORE": 0.5,        
                "TREND_SENSITIVITY": 0.9,  
                "POSITION_RISK_FACTOR": 1.0,  
                "MAX_TRADE_SIZE": 3,       
                "MIN_SPREAD": 5.0          
            }
        }
    
    def get_best_prices(self, order_depth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None, None
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return best_bid, best_ask
    
    def update_price_history(self, state):
        for product, depth in state.order_depths.items():
            if product not in self.price_history:
                self.price_history[product] = []
                
            best_prices = self.get_best_prices(depth)
            if None not in best_prices:
                best_bid, best_ask = best_prices
                mid_price = (best_bid + best_ask) / 2
                self.price_history[product].append(mid_price)
                if len(self.price_history[product]) > 100:
                    self.price_history[product].pop(0)
                
                self.last_midprices[product] = mid_price
                if len(self.price_history[product]) > 15:
                    recent_avg = np.mean(self.price_history[product][-5:])
                    older_avg = np.mean(self.price_history[product][-15:-5])
                    self.price_trends[product] = recent_avg - older_avg
        
        if Product.PICNIC_BASKET1 in self.last_midprices and Product.PICNIC_BASKET2 in self.last_midprices:
            spread = self.last_midprices[Product.PICNIC_BASKET1] - self.last_midprices[Product.PICNIC_BASKET2]
            self.basket_spread_history.append(spread)
            if len(self.basket_spread_history) > 50:
                self.basket_spread_history.pop(0)
    
    def calculate_fair_values(self, state):
        """Calculate fair values based on component prices with enhanced weighting"""
        components = {}
        fair_values = {}
        for product in [Product.CROISSANTS, Product.JAMS, Product.DJEMBES]:
            if product in state.order_depths:
                best_prices = self.get_best_prices(state.order_depths[product])
                if None not in best_prices:
                    best_bid, best_ask = best_prices
                    best_bid_vol = abs(state.order_depths[product].buy_orders[best_bid])
                    best_ask_vol = abs(state.order_depths[product].sell_orders[best_ask])
                    total_volume = best_bid_vol + best_ask_vol
                    if total_volume > 0:
                        vwap_mid = (best_bid * best_ask_vol + best_ask * best_bid_vol) / total_volume
                        components[product] = vwap_mid
                    else:
                        components[product] = (best_bid + best_ask) / 2

        for product in [Product.CROISSANTS, Product.JAMS, Product.DJEMBES]:
            if product not in components and product in self.price_history and len(self.price_history[product]) > 0:
                price_history = np.array(self.price_history[product])
                if len(price_history) > 30:
                    if abs(self.price_trends[product]) > 0.1:
                        weights = np.exp(np.linspace(0, 1, 30))
                        weights = weights / sum(weights)
                        recent_history = price_history[-30:]
                        components[product] = np.sum(recent_history * weights)
                    else:
                        components[product] = np.median(self.price_history[product][-20:])
                else:
                    components[product] = np.median(self.price_history[product])

        if all(product in components for product in [Product.CROISSANTS, Product.JAMS, Product.DJEMBES]):
            fair_values[Product.PICNIC_BASKET2] = 4 * components[Product.CROISSANTS] + 2 * components[Product.JAMS]
            fair_values[Product.PICNIC_BASKET1] = 1.5 * fair_values[Product.PICNIC_BASKET2] + (2/3) * components[Product.DJEMBES]
            fair_values.update(components)
            self.fair_values = fair_values

        return fair_values
 
    def get_swmid(self, order_depth) -> float:
        """Calculate the volume-weighted mid price"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        
        
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )

    def get_synthetic_basket1_order_depth(self, state: TradingState) -> OrderDepth:
        """Create synthetic order depth for basket1 from its components"""
        JAM_PER_BASKET1 = 3
        DJEMBE_PER_BASKET1 = 1
        CROISSANT_PER_BASKET1 = 6
        
        synthetic_order_depth = OrderDepth()
        order_depths = state.order_depths        
        if not all(product in order_depths for product in [Product.CROISSANTS, Product.JAMS, Product.DJEMBES]):
            return synthetic_order_depth
        
        CROISSANT_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        CROISSANT_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )
        JAM_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        JAM_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )
        DJEMBE_best_bid = (
            max(order_depths[Product.DJEMBES].buy_orders.keys())
            if order_depths[Product.DJEMBES].buy_orders
            else 0
        )
        DJEMBE_best_ask = (
            min(order_depths[Product.DJEMBES].sell_orders.keys())
            if order_depths[Product.DJEMBES].sell_orders
            else float("inf")
        )
        
        implied_bid = (
            CROISSANT_best_bid * CROISSANT_PER_BASKET1
            + JAM_best_bid * JAM_PER_BASKET1
            + DJEMBE_best_bid * DJEMBE_PER_BASKET1
        )
        implied_ask = (
            CROISSANT_best_ask * CROISSANT_PER_BASKET1
            + JAM_best_ask * JAM_PER_BASKET1
            + DJEMBE_best_ask * DJEMBE_PER_BASKET1
        )
        
        if implied_bid > 0 and CROISSANT_best_bid > 0 and JAM_best_bid > 0 and DJEMBE_best_bid > 0:
            CROISSANT_bid_volume = abs(order_depths[Product.CROISSANTS].buy_orders[CROISSANT_best_bid]) // CROISSANT_PER_BASKET1
            JAM_bid_volume = abs(order_depths[Product.JAMS].buy_orders[JAM_best_bid]) // JAM_PER_BASKET1
            DJEMBE_bid_volume = abs(order_depths[Product.DJEMBES].buy_orders[DJEMBE_best_bid]) // DJEMBE_PER_BASKET1
            implied_bid_volume = min(CROISSANT_bid_volume, JAM_bid_volume, DJEMBE_bid_volume)
            
            if implied_bid_volume > 0:
                synthetic_order_depth.buy_orders[implied_bid] = implied_bid_volume
        
        if implied_ask < float("inf") and CROISSANT_best_ask < float("inf") and JAM_best_ask < float("inf") and DJEMBE_best_ask < float("inf"):
            CROISSANT_ask_volume = abs(order_depths[Product.CROISSANTS].sell_orders[CROISSANT_best_ask]) // CROISSANT_PER_BASKET1
            JAM_ask_volume = abs(order_depths[Product.JAMS].sell_orders[JAM_best_ask]) // JAM_PER_BASKET1
            DJEMBE_ask_volume = abs(order_depths[Product.DJEMBES].sell_orders[DJEMBE_best_ask]) // DJEMBE_PER_BASKET1
            implied_ask_volume = min(CROISSANT_ask_volume, JAM_ask_volume, DJEMBE_ask_volume)
            
            if implied_ask_volume > 0:
                synthetic_order_depth.sell_orders[implied_ask] = -implied_ask_volume
        
        return synthetic_order_depth

    def get_synthetic_basket2_order_depth(self, state) -> OrderDepth:
        """Create synthetic order depth for basket2 from its components"""
        
        CROISSANT_PER_BASKET2 = 4
        JAM_PER_BASKET2 = 2
        
        synthetic_order_depth = OrderDepth()
        order_depths = state.order_depths
        
        if not all(product in order_depths for product in [Product.CROISSANTS, Product.JAMS]):
            return synthetic_order_depth
        
        CROISSANT_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        CROISSANT_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )
        JAM_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        JAM_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )
        
        implied_bid = (
            CROISSANT_best_bid * CROISSANT_PER_BASKET2
            + JAM_best_bid * JAM_PER_BASKET2
        )
        implied_ask = (
            CROISSANT_best_ask * CROISSANT_PER_BASKET2
            + JAM_best_ask * JAM_PER_BASKET2
        )
        
        if implied_bid > 0 and CROISSANT_best_bid > 0 and JAM_best_bid > 0:
            CROISSANT_bid_volume = abs(order_depths[Product.CROISSANTS].buy_orders[CROISSANT_best_bid]) // CROISSANT_PER_BASKET2
            JAM_bid_volume = abs(order_depths[Product.JAMS].buy_orders[JAM_best_bid]) // JAM_PER_BASKET2
            implied_bid_volume = min(CROISSANT_bid_volume, JAM_bid_volume)
            
            if implied_bid_volume > 0:
                synthetic_order_depth.buy_orders[implied_bid] = implied_bid_volume
        
        if implied_ask < float("inf") and CROISSANT_best_ask < float("inf") and JAM_best_ask < float("inf"):
            CROISSANT_ask_volume = abs(order_depths[Product.CROISSANTS].sell_orders[CROISSANT_best_ask]) // CROISSANT_PER_BASKET2
            JAM_ask_volume = abs(order_depths[Product.JAMS].sell_orders[JAM_best_ask]) // JAM_PER_BASKET2
            implied_ask_volume = min(CROISSANT_ask_volume, JAM_ask_volume)
            
            if implied_ask_volume > 0:
                synthetic_order_depth.sell_orders[implied_ask] = -implied_ask_volume
        
        return synthetic_order_depth

    def convert_synthetic_basket2_orders(self, synthetic_orders, state):
        """Convert synthetic basket2 orders to component orders"""
        
        component_orders = {
            Product.CROISSANTS: [],
            Product.JAMS: [],
        }
        
        
        order_depths = state.order_depths
        
        
        synthetic_basket2_order_depth = self.get_synthetic_basket2_order_depth(state)
        
        
        best_bid = (
            max(synthetic_basket2_order_depth.buy_orders.keys())
            if synthetic_basket2_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket2_order_depth.sell_orders.keys())
            if synthetic_basket2_order_depth.sell_orders
            else float("inf")
        )
        
        
        for order in synthetic_orders:
            
            price = order.price
            quantity = order.quantity
            
            
            if quantity > 0 and price >= best_ask:
                
                CROISSANT_price = min(order_depths[Product.CROISSANTS].sell_orders.keys())
                JAM_price = min(order_depths[Product.JAMS].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                
                CROISSANT_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                JAM_price = max(order_depths[Product.JAMS].buy_orders.keys())
            else:
                
                continue
            
            
            CROISSANT_order = Order(
                Product.CROISSANTS,
                CROISSANT_price,
                quantity * self.BASKET2_COMPOSITION[Product.CROISSANTS],
            )
            JAM_order = Order(
                Product.JAMS,
                JAM_price,
                quantity * self.BASKET2_COMPOSITION[Product.JAMS],
            )
            
            
            component_orders[Product.CROISSANTS].append(CROISSANT_order)
            component_orders[Product.JAMS].append(JAM_order)
        
        return component_orders

    def execute_basket1_spread_orders(self, target_position, basket_position, state):
        """Execute spread orders between basket1 and synthetic basket1"""
        if target_position == basket_position:
            return None
        
        target_quantity = abs(target_position - basket_position)
        order_depths = state.order_depths
        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket1_order_depth(state)
        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])
            
            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(synthetic_order_depth.buy_orders[synthetic_bid_price])
            
            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)
    
            return [ Order(Product.PICNIC_BASKET1, basket_ask_price, execute_volume) ]
        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])
            
            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(synthetic_order_depth.sell_orders[synthetic_ask_price])
            
            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            
            return [ Order(Product.PICNIC_BASKET1, basket_bid_price, -execute_volume) ]

    def execute_basket2_spread_orders(self, target_position, basket_position, state):
        """Execute spread orders between basket2 and synthetic basket2"""
        if target_position == basket_position:
            return None
        
        target_quantity = abs(target_position - basket_position)
        order_depths = state.order_depths
        basket_order_depth = order_depths[Product.PICNIC_BASKET2]
        synthetic_order_depth = self.get_synthetic_basket2_order_depth(state)
        
        if target_position > basket_position:
            
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])
            
            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(synthetic_order_depth.buy_orders[synthetic_bid_price])
            
            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            
            basket_orders = [ Order(Product.PICNIC_BASKET2, basket_ask_price, execute_volume) ]
            synthetic_orders = [ Order("SYNTHETIC2", synthetic_bid_price, -execute_volume) ]
            aggregate_orders = self.convert_synthetic_basket2_orders(synthetic_orders, state)
            aggregate_orders[Product.PICNIC_BASKET2] = basket_orders

            return aggregate_orders
        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])
            
            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(synthetic_order_depth.sell_orders[synthetic_ask_price])
            
            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            
            basket_orders = [ Order(Product.PICNIC_BASKET2, basket_bid_price, -execute_volume) ]
            synthetic_orders = [ Order("SYNTHETIC2", synthetic_ask_price, execute_volume) ]
            aggregate_orders = self.convert_synthetic_basket2_orders(synthetic_orders, state)
            aggregate_orders[Product.PICNIC_BASKET2] = basket_orders

            return aggregate_orders

    def basket1_pair_trading(self, state: TradingState):
        """Pair trading between basket1 and synthetic basket1"""
        if not hasattr(self, "basket1_spread_data"):
            self.basket1_spread_data = {
                "spread_history": [],
                "prev_zscore": 0,
                "default_spread_mean":52.633,  
                "zscore_threshold":2.5,   
                "spread_std_window": 50,   
                "target_position": 60        
            }
        
        if Product.PICNIC_BASKET1 not in state.order_depths:
            return {}
        
        basket_position = state.position.get(Product.PICNIC_BASKET1, 0)
        basket_order_depth = state.order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket1_order_depth(state)
    
        if not basket_order_depth.buy_orders or not basket_order_depth.sell_orders:
            return {}
        if not synthetic_order_depth.buy_orders or not synthetic_order_depth.sell_orders:
            return {}
        
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        if basket_swmid is None or synthetic_swmid is None:
            return {}
        
        spread = basket_swmid - synthetic_swmid

        self.basket1_spread_data["spread_history"].append(spread)
        if len(self.basket1_spread_data["spread_history"]) < self.basket1_spread_data["spread_std_window"]:
            return {}
        elif len(self.basket1_spread_data["spread_history"]) > self.basket1_spread_data["spread_std_window"]:
            self.basket1_spread_data["spread_history"].pop(0)
        spread_std = np.std(self.basket1_spread_data["spread_history"])

        spread_std = 82.33
        zscore = (spread - self.basket1_spread_data["default_spread_mean"]) / max(spread_std, 0.1)
        
        orders = {}
        if zscore >= self.basket1_spread_data["zscore_threshold"]:
            if basket_position != -self.basket1_spread_data["target_position"]:
                result = self.execute_basket1_spread_orders(
                    -self.basket1_spread_data["target_position"],
                    basket_position,
                    state
                )
                if result:
                    orders.update(result)
        elif zscore <= -self.basket1_spread_data["zscore_threshold"]:
            if basket_position != self.basket1_spread_data["target_position"]:
                result = self.execute_basket1_spread_orders(
                    self.basket1_spread_data["target_position"],
                    basket_position,
                    state
                )
                if result:
                    orders.update(result)
        
        self.basket1_spread_data["prev_zscore"] = zscore
        
        return orders

    def basket2_pair_trading(self, state):
        """Pair trading between basket2 and synthetic basket2"""
        if not hasattr(self, "basket2_spread_data"):
            self.basket2_spread_data = {
                "spread_history": [],
                "prev_zscore": 0,
                "default_spread_mean": 23.33,
                "zscore_threshold": 1.65,
                "spread_std_window": 40,
                "target_position": 100,
                "zscore_return_threshold":0.95,
                "exit_fraction": 0.24,
                "active_side": None,
                "zscore_stop_loss": 2.5
            }

        if Product.PICNIC_BASKET2 not in state.order_depths:
            return {}

        basket_position = state.position.get(Product.PICNIC_BASKET2, 0)
        basket_order_depth = state.order_depths[Product.PICNIC_BASKET2]
        synthetic_order_depth = self.get_synthetic_basket2_order_depth(state)

        if not basket_order_depth.buy_orders or not basket_order_depth.sell_orders:
            return {}
        if not synthetic_order_depth.buy_orders or not synthetic_order_depth.sell_orders:
            return {}

        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        if basket_swmid is None or synthetic_swmid is None:
            return {}

        spread = basket_swmid - synthetic_swmid
        hist = self.basket2_spread_data["spread_history"]
        hist.append(spread)
        if len(hist) < self.basket2_spread_data["spread_std_window"]:
            return {}
        if len(hist) > self.basket2_spread_data["spread_std_window"]:
            hist.pop(0)

        spread_std = 54.84  
        zscore = (spread - self.basket2_spread_data["default_spread_mean"]) / max(spread_std, 0.1)

        orders = {}
        if zscore >= self.basket2_spread_data["zscore_threshold"]:
            if basket_position > -self.basket2_spread_data["target_position"]:
                res = self.execute_basket2_spread_orders(
                    -self.basket2_spread_data["target_position"],
                    basket_position,
                    state
                )
                if res:
                    orders.update(res)
                    self.entry_info[Product.PICNIC_BASKET2] = {
                        "price": basket_swmid,
                        "side": "short"
                    }
                    self.basket2_spread_data["active_side"] = "short"
                    return orders
        elif zscore <= -self.basket2_spread_data["zscore_threshold"]:
            if basket_position < self.basket2_spread_data["target_position"]:
                res = self.execute_basket2_spread_orders(
                    self.basket2_spread_data["target_position"],
                    basket_position,
                    state
                )
                if res:
                    orders.update(res)
                    self.entry_info[Product.PICNIC_BASKET2] = {
                        "price": basket_swmid,
                        "side": "long"
                    }
                    self.basket2_spread_data["active_side"] = "long"
                    return orders

        self.basket2_spread_data["prev_zscore"] = zscore
        return orders

    def generate_orders(self, state, trader_data=None):
        self.update_price_history(state)
        self.calculate_fair_values(state)
        all_orders = {}

        basket1_pair_orders = self.basket1_pair_trading(state)
        if basket1_pair_orders:
            all_orders.update(basket1_pair_orders)

        basket2_pair_orders = self.basket2_pair_trading(state)
        if basket2_pair_orders:
            all_orders.update(basket2_pair_orders)

        return all_orders

class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> tuple[list[Order], int]:
        self.orders = []
        self.conversions = 0

        self.act(state)

        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def convert(self, amount: int) -> None:
        self.conversions += amount
    
    def save(self):
        pass

    def load(self):
        pass

class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

        self.window = deque()
        self.window_size = 10

    @abstractmethod
    def get_true_value(self, state: TradingState) -> int:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()

        soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size / 2 and self.window[-1]
        hard_liquidate = len(self.window) == self.window_size and all(self.window)

        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < self.limit * -0.5 else true_value

        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            self.buy(true_value, quantity)
            to_buy -= quantity

        if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            self.buy(true_value - 2, quantity)
            to_buy -= quantity

        if to_buy > 0:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy_price, popular_buy_price + 1)
            self.buy(price, to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            self.sell(true_value, quantity)
            to_sell -= quantity

        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            self.sell(true_value + 2, quantity)
            to_sell -= quantity

        if to_sell > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)

    def save(self) -> JSON:
        return list(self.window)

    def load(self, data: JSON) -> None:
        self.window = deque(data)

class VolcanicRockStrategy(MarketMakingStrategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.prices = deque(maxlen=20)
        self.volatility = 0.2  

    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        best_bid = buy_orders[0][0] if buy_orders else 0
        best_ask = sell_orders[0][0] if sell_orders else INF
        mid_price = (best_bid + best_ask) / 2
        
        self.prices.append(mid_price)
        if len(self.prices) > 5:
            returns = [math.log(self.prices[i] / self.prices[i-1]) 
                      for i in range(1, len(self.prices))]
            if returns:
                self.volatility = max(0.1, (sum([r*r for r in returns]) / len(returns))**0.5 * math.sqrt(365))

        return int(mid_price)

    def save(self) -> JSON:
        return {
            "window": list(self.window),
            "prices": list(self.prices),
            "volatility": self.volatility
        }

    def load(self, data: JSON) -> None:
        if isinstance(data, dict):
            self.window = deque(data.get("window", []))
            self.prices = deque(data.get("prices", []), maxlen=20)
            self.volatility = data.get("volatility", 0.20)
        else:
            self.window = deque(data) if data else deque()
            
class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        if volatility <= 0 or time_to_expiry <= 0:
            return max(0, spot - strike)
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * normalDist.cdf(d1) - strike * exp(-0 * time_to_expiry) * normalDist.cdf(d2)  
        return call_price

    @staticmethod
    def implied_volatility(target_price, spot, strike, time_to_expiry, initial_guess=0.20, max_iterations=100, tolerance=1e-5):
        if time_to_expiry <= 0:
            return 0 

        volatility = initial_guess
        for i in range(max_iterations):
            try:
                price = BlackScholes.black_scholes_call(spot, strike, time_to_expiry, volatility)
                d1 = (log(spot / strike) + (0.5 * volatility ** 2) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
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
                if 'price' in locals() and price < target_price:
                    volatility += 0.01
                else:
                    volatility -= 0.01
                volatility = max(0.0001, min(volatility, 2.0))
        return volatility

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        
        self.LIMIT = {
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
            Product.VOLCANIC_ROCK_VOUCHER_10500: 200
        }
        
        self.kelp_prices = []
        self.kelp_vwap = []
        self.squid_prices = []
        self.squid_vwap = []
        self.historical_data: Dict[str, Dict] = {}
        self.picnic_strategy = PicnicBasketStrategy()

    def _get_mid_price(self, product: Symbol, order_depth: OrderDepth) -> float | None:
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            return (best_ask + best_bid) / 2.0
        return None  

    def _get_effective_voucher_price(self, voucher_product: Symbol, order_depth: OrderDepth, traderData: Dict) -> float | None:
        mid_price = self._get_mid_price(voucher_product, order_depth)
        if mid_price is not None:
            traderData[voucher_product]["last_price"] = mid_price
            return mid_price
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                if voucher_product in traderData and "last_price" in traderData[voucher_product]:
                    return traderData[voucher_product]["last_price"]
                return (best_bid + best_ask) / 2.0
            else:
                traderData[voucher_product]["last_price"] = best_bid
                return best_bid
        elif order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            traderData[voucher_product]["last_price"] = best_ask
            return best_ask
        elif voucher_product in traderData and "last_price" in traderData[voucher_product]:
            return traderData[voucher_product]["last_price"]
        return None
    
    def calculate_mean(self, data: List[float]) -> float:
        """Calculates the mean of a list of numbers."""
        if not data:
            return 0
        return sum(data) / len(data)
    
    def calculate_std_dev(self, data: List[float], mean: float) -> float:
        """Calculates the standard deviation of a list of numbers."""
        if len(data) < 2:
            return 0
        variance = sum([(x - mean) ** 2 for x in data]) / (len(data) - 1)
        return variance ** 0.5
    
    def squid_ink_strategy(self, mid_price: float, best_bid: float, best_ask: float, position: int) -> List[Order]:
        product = 'SQUID_INK'

        
        if product not in self.historical_data:
            self.historical_data[product] = {
                'prices': [],
                'last_ema': None 
            }

        
        self.historical_data[product]['prices'].append(mid_price)
        prices = self.historical_data[product]['prices']
        
        last_ema = self.historical_data[product].get('last_ema')

        
        params = self.params.get(product)
        if not params:
            print(f"Warning: Parameters for {product} not found.")
            return []

        window_size = params.get("window_size", 60)
        ema_alpha = params.get("ema_alpha")
        z_score_threshold = params.get("z_score_threshold", 1.1)
        trade_size = params.get("trade_size", 6)
        position_limit = params.get("position_limit", 50)

        
        if ema_alpha is None:
            alpha = 2 / (window_size + 1)
        else:
            alpha = ema_alpha

        
        current_ema = None
        num_prices = len(prices)

        if num_prices < window_size:
            
            return []
        elif num_prices == window_size:
            
            
            
            current_ema = self.calculate_mean(prices)
            print(f"DEBUG: {product} Initializing EMA. Using SMA of first {window_size} prices: {current_ema:.2f}")
        else: 
            if last_ema is None:
                
                
                print(f"Warning: {product} last_ema is None despite sufficient data. Re-initializing with SMA.")
                
                last_ema = self.calculate_mean(prices[-(window_size+1):-1])

            
            
            current_ema = alpha * mid_price + (1 - alpha) * last_ema

        
        
        if current_ema is not None:
             self.historical_data[product]['last_ema'] = current_ema
        else:
             
             return []
      
        
        
        recent_prices = prices[-window_size:]
        mean_price_for_stddev = self.calculate_mean(recent_prices)
        std_dev_price = self.calculate_std_dev(recent_prices, mean_price_for_stddev)

        if std_dev_price == 0:
            z_score = 0
        else:
            
            z_score = (mid_price - current_ema) / std_dev_price

        
        orders: List[Order] = []
        reference_price = current_ema 

        
        if z_score < -z_score_threshold and position < position_limit:
            buy_volume = min(trade_size, position_limit - position)
            if buy_volume > 0:
                buy_price = int(min(mid_price, best_bid + 1)) 
                print(f"INFO: {product} BUY Signal. Z: {z_score:.2f}, EMA: {current_ema:.2f}, Mid: {mid_price:.2f}, Pos: {position}. Placing order for {buy_volume} @ {buy_price}")
                orders.append(Order(product, buy_price, buy_volume))

        
        elif z_score > z_score_threshold and position > -position_limit:
            sell_volume = min(trade_size, position_limit + position)
            if sell_volume > 0:
                sell_price = int(max(mid_price, best_ask - 1)) 
                print(f"INFO: {product} SELL Signal. Z: {z_score:.2f}, EMA: {current_ema:.2f}, Mid: {mid_price:.2f}, Pos: {position}. Placing order for {-sell_volume} @ {sell_price}")
                orders.append(Order(product, sell_price, -sell_volume))

        return orders

    def volcanic_voucher_orders(
        self,
        voucher_product: Symbol,
        voucher_order_depth: OrderDepth,
        voucher_position: int,
        voucher_limit: int,
        voucher_params: Dict,
        voucher_traderData: Dict,  
        implied_volatility: float,
    ) -> List[Order]:
        orders: List[Order] = []
        if implied_volatility is None:
            return orders

        
        voucher_traderData['past_coupon_vol'].append(implied_volatility)
        if len(voucher_traderData['past_coupon_vol']) > voucher_params['std_window']:
            voucher_traderData['past_coupon_vol'].pop(0)

        if len(voucher_traderData['past_coupon_vol']) < voucher_params['std_window'] / 2:
            return orders

        current_vols = voucher_traderData['past_coupon_vol']
        mean_vol = np.mean(current_vols)
        std_dev_vol = np.std(current_vols)
        if std_dev_vol < 1e-5:
            z_score = 0
        else:
            z_score = (implied_volatility - mean_vol) / std_dev_vol

        desired_trade_quantity = 0
        if z_score >= voucher_params['zscore_threshold']:
            desired_trade_quantity = -voucher_limit - voucher_position  
        elif z_score <= -voucher_params['zscore_threshold']:
            desired_trade_quantity = voucher_limit - voucher_position  

        if desired_trade_quantity == 0:
            return orders

        if desired_trade_quantity < 0:
            if voucher_order_depth.buy_orders:
                best_bid = max(voucher_order_depth.buy_orders.keys())
                available_at_bid = voucher_order_depth.buy_orders[best_bid]
                qty_to_sell = min(abs(desired_trade_quantity), available_at_bid, voucher_limit + voucher_position)
                if qty_to_sell > 0:
                    orders.append(Order(voucher_product, best_bid, -qty_to_sell))
        elif desired_trade_quantity > 0:
            if voucher_order_depth.sell_orders:
                best_ask = min(voucher_order_depth.sell_orders.keys())
                available_at_ask = abs(voucher_order_depth.sell_orders[best_ask])
                qty_to_buy = min(desired_trade_quantity, available_at_ask, voucher_limit - voucher_position)
                if qty_to_buy > 0:
                    orders.append(Order(voucher_product, best_ask, qty_to_buy))
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
            buy_order_volume, sell_order_volume, fair_value, 1)

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(Product.RAINFOREST_RESIN, int(round(bbbf + 1)), buy_quantity))  

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(Product.RAINFOREST_RESIN, int(round(baaf - 1)), -sell_quantity))  

        return orders
    
    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int, position_limit: int,
        product: str, buy_order_volume: int, sell_order_volume: int, fair_value: float
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
            
            fair_value = (mm_ask + mm_bid) / 2

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

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}  

        traderData = {}
        kelpData = {}
        if state.traderData is not None and state.traderData != "":
            try:
                traderData = jsonpickle.decode(state.traderData)
            except Exception as e:
                print(f"Error decoding traderData: {e}")
                traderData = {}

        if Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            resin_orders_list = self.resin_orders(state.order_depths[Product.RAINFOREST_RESIN], 10000, resin_position, self.LIMIT[Product.RAINFOREST_RESIN])
            result[Product.RAINFOREST_RESIN] = resin_orders_list

        if Product.KELP in state.order_depths:
            kelp_position = state.position.get(Product.KELP, 0)
            result[Product.KELP] = self.kelp_orders(state.order_depths[Product.KELP], 10, 1, kelp_position, self.LIMIT[Product.KELP])

        if Product.SQUID_INK in state.order_depths:
            order_depth = state.order_depths[Product.SQUID_INK]
            result[Product.SQUID_INK] = []
            if order_depth.sell_orders and order_depth.buy_orders:
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                mid_price = (best_ask + best_bid) / 2
                current_position = state.position.get(Product.SQUID_INK, 0)
                result[Product.SQUID_INK] = self.squid_ink_strategy(mid_price, best_bid, best_ask, current_position)

        picnic_orders = self.picnic_strategy.generate_orders(state, traderData)
        result.update(picnic_orders)

        if Product.VOLCANIC_ROCK in state.order_depths:
            symbol = Product.VOLCANIC_ROCK
            strategy = VolcanicRockStrategy(Product.VOLCANIC_ROCK, self.LIMIT[Product.VOLCANIC_ROCK])
            if symbol in traderData:
                strategy.load(traderData.get(symbol))
            if symbol in state.order_depths:
                strategy_orders, _ = strategy.run(state)
                result[symbol] = strategy_orders
            traderData[symbol] = strategy.save()
            
        for voucher_product in VOLCANIC_VOUCHERS_Trade:
            if voucher_product not in traderData:
                traderData[voucher_product] = {"past_coupon_vol": [], "last_price": None}
            elif "past_coupon_vol" not in traderData[voucher_product]:
                traderData[voucher_product]["past_coupon_vol"] = []

        rock_mid_price = None
        if Product.VOLCANIC_ROCK in state.order_depths:
            rock_order_depth = state.order_depths[Product.VOLCANIC_ROCK]
            rock_mid_price = self._get_mid_price(Product.VOLCANIC_ROCK, rock_order_depth)
        if rock_mid_price is None:
            print("Warning: Could not determine Volcanic Rock mid-price. Skipping voucher logic.")
            return result

        for voucher_product in VOLCANIC_VOUCHERS_Trade:
            if voucher_product in state.order_depths:
                voucher_order_depth = state.order_depths[voucher_product]
                voucher_params = PARAMS[voucher_product]
                voucher_limit = self.LIMIT[voucher_product]
                voucher_position = state.position.get(voucher_product, 0)
                voucher_traderData = traderData[voucher_product]
                
                voucher_price = self._get_effective_voucher_price(voucher_product, voucher_order_depth, traderData)
                if voucher_price is not None and voucher_price > 0:
                    tte = 4/365 - (state.timestamp/1000000)/365
                    implied_volatility = None
                    if tte > 1e-6:
                        try:
                            implied_volatility = BlackScholes.implied_volatility(
                                target_price=voucher_price,
                                spot=rock_mid_price,
                                strike=voucher_params["strike"],
                                time_to_expiry=tte
                            )
                        except Exception as e:
                            print(f"Error calculating IV for {voucher_product}: {e}")
                    if implied_volatility is not None:
                        current_voucher_orders = self.volcanic_voucher_orders(
                            voucher_product=voucher_product,
                            voucher_order_depth=voucher_order_depth,
                            voucher_position=voucher_position,
                            voucher_limit=voucher_limit,
                            voucher_params=voucher_params,
                            voucher_traderData=voucher_traderData,
                            implied_volatility=implied_volatility,
                        )
                        if voucher_product not in result:
                            result[voucher_product] = []
                        result[voucher_product].extend(current_voucher_orders)
        
        kelpData = {
            "kelp_prices": self.kelp_prices,
            "kelp_vwap": self.kelp_vwap, 
        }

        traderData.update(kelpData)
        conversions = 0  
        traderData = jsonpickle.encode(traderData, unpicklable=False)
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData
