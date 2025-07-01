# region imports
from AlgorithmImports import *
from datetime import datetime, timedelta
import gym
import random
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import math
import torch, numpy as np, random  

# endregion

class MuscularVioletCaribou(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2024, 10, 5)
        # date = datetime(2024, 10, 2, 14, 0)
        self.set_cash(500000)
        self.step_counter = 0
        self.learn_every = 750

        self.training_end_date = datetime(2025, 2, 27)   # last date of training phase
        self.set_end_date(2025, 5, 4)           # end of full backtest
        self.equity_str = "XLK"
        #All assets being traded in a Rama DS
        self.sector_map = {
            # ("XLE", False): [
            #     (Futures.Energy.CRUDE_OIL_WTI, False),
            #     (Futures.Energy.NATURAL_GAS, False)
            # ]
            # ("XLI", False): [
            #     #(Futures.Indices.DOW_30_E_MINI, False),
            #     (Futures.Metals.COPPER, False)
            # ],
            # ("XLV", False): [
            #     (Futures.Indices.SP_500_E_MINI, False),
            #     (Futures.Indices.VIX, False)
            # ]
            #  ("XLF", False): [
            #      (Futures.Financials.Y_10_TREASURY_NOTE, False),
            #      (Futures.Financials.EURO_DOLLAR, False)
            # ],
            # ("XLY", False): [
            #     (Futures.Indices.SP_500_E_MINI, False),
            #     (Futures.Indices.NASDAQ_100_E_MINI, False)
            # ],
            # ("XLP", False): [
            #     (Futures.Indices.SP_500_E_MINI, False),
            #     (Futures.Financials.Y_10_TREASURY_NOTE, False)
            # ],
            # ("XLB", False): [
            #     #(Futures.Metals.COPPER, False),
            #     #(Futures.Metals.GOLD, False),
            #     (Futures.Metals.US_MIDWEST_DOMESTIC_HOT_ROLLED_COIL_STEEL_CRU_INDEX, False)
            # ],
            # ("XLU", False): [
            #     (Futures.Financials.Y_10_TREASURY_NOTE, False),
            #     (Futures.Financials.Y_30_TREASURY_BOND, False)
            # ],
            # ("XLC", False): [
            #     (Futures.Indices.NASDAQ_100_E_MINI, False)
            # ],
            ("XLK", False): [
                #(Futures.Indices.NASDAQ_100_BIOTECHNOLOGY_E_MINI, False)
                (Futures.Indices.NASDAQ_100_E_MINI, False)
            ]
            # ("XLRE", False): [
            #     (Futures.Financials.Y_10_TREASURY_NOTE, False)
            #     #(Futures.Indices.DOW_JONES_REAL_ESTATE, False)
            # ]
        }
        self.macd_map = {}
        self.boll_map = {}
        self.stop_loss_pct = 0.2


        #Loop and add assets
        self.equity_symbols = {}
        self.option_symbols = {}
        self.future_symbols = {}

        # option = self.add_option("XLE", Resolution.HOUR)

        for etf, b in self.sector_map.keys():
            # Add Equity
            equity = self.add_equity(etf, Resolution.HOUR)
            self.equity_symbols[etf] = equity.symbol
            
            # Add Option
            option = self.add_option(etf, Resolution.HOUR)
            option.set_filter(-5, +5, 30, 60)
            self.option_symbols[etf] = option.symbol

            # Add Futures (can be multiple per ETF)
            for future_name, __ in self.sector_map[(etf, b)]:
                future = self.add_future(future_name, Resolution.HOUR)
                future.set_filter(lambda x: x.front_month())
                self.future_symbols[future_name] = future.symbol

        
        for etf, _ in self.sector_map.keys():
            self.macd_map[etf] = self.macd(self.equity_symbols[etf], 12, 26, 9, MovingAverageType.WILDERS, Resolution.HOUR)
            self.boll_map[etf] = self.bb(self.equity_symbols[etf], 20, 2, MovingAverageType.SIMPLE, Resolution.HOUR)
        #Initialize stable baselines model
        # seed = 42  
        # random.seed(seed)  
        # np.random.seed(seed)  
        # torch.manual_seed(seed)  
        self.env = RLTradingAgent(self)
        self.model = PPO("MlpPolicy", self.env, verbose=1) #Toggle params as need be

    def liquidate_expiring_positions(self):
        for symbol, holding in self.portfolio.items():
            if not holding.invested:
                continue

            security_type = symbol.security_type
            contract_expiry = symbol.ID.date.date()
            days_to_expiry = (contract_expiry - self.time.date()).days
            unrealized_profit = holding.unrealized_profit
            entry_price = self.portfolio[symbol].average_price
            unrealized_pct = unrealized_profit / (entry_price * holding.quantity)

            # if unrealized_pct <= -self.stop_loss_pct:
            #     self.log(f"Liquidating {symbol}: -10% stop loss hit")
            #     self.liquidate(symbol)
            #     continue

            should_liquidate = False
            reason = ""

            if security_type in [SecurityType.OPTION, SecurityType.FUTURE]:
                if days_to_expiry <= 2:
                    should_liquidate = True
                    reason = f"expiring soon ({days_to_expiry} days left)"
                elif unrealized_profit > 0:
                    should_liquidate = True
                    reason = f"in profit (${unrealized_profit:.2f})"

                if should_liquidate:
                    self.log(f"Liquidating {symbol.Value} ({security_type}) - Reason: {reason}")
                    self.liquidate(symbol)


    def on_data(self, data: Slice):
        # Skip if environment can't generate observations
        self.liquidate_expiring_positions()
        self.env.collect_state(data)
        self.step_counter += 1

        # TRAINING PHASE
        if self.time < self.training_end_date:
            obs, reward, done, info = self.env.step(data)

            if self.step_counter % self.learn_every == 0:
                self.model.learn(total_timesteps=self.learn_every, reset_num_timesteps=False)
                self.model.save(f"ppo_checkpoint_step_{self.step_counter}")

        # DEPLOYMENT PHASE (after training ends)
        else:
            self.env.run_live(data)  # Predict and trade using trained model

        # Save final model at end of backtest
        if self.time.date() == self.end_date.date():
            self.model.save("ppo_final_trader")
    
    def on_order_event(self, order_event):
        self.log(str(order_event))

class RLTradingAgent(gym.Env):
    def __init__(self, algorithm: MuscularVioletCaribou):
        super().__init__()
        self.algorithm = algorithm
        self.current_obs = {}
        self.contract_dict = {}
        self.contract = None
        self.call = None
        self.put = None
        self.position = 0
        self.prev_portfolio_val = None
        self.reward_history = []
        self.obs_history = []
        self.cur_slice = None
        self.sector_map = algorithm.sector_map
        self.macd_map = algorithm.macd_map
        self.boll_map = algorithm.boll_map
        self.scale_factor = 100
        self.trailing_stop_prices = {}  # Dictionary to store stop prices per contract
        self.stop_loss_pct = 0.10 
        # Define observation and action spaces
        self.observation_space = self._define_observation_space()
        self.action_space = self._define_action_space()


        self.equity_symbols = algorithm.equity_symbols
        self.option_symbols = algorithm.option_symbols
        self.future_symbols = algorithm.future_symbols
    

    def _define_observation_space(self):
        # Assuming each observation is a 10-dimensional float32 vector
        return spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)

    def _define_action_space(self):
        return spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell


    def reset(self):
        """
        Reset the environment at the start of a new episode.
        """
        self.reward_history = []
        self.obs_history = []
        self.position = 0
        self.prev_portfolio_val = None
        return self.current_obs[self.algorithm.equity_str]

    def collect_state(self, slice: Slice): #Come back with math to fix logic behind picking an option
        """
        Collect observation from the current backtest slice.
        Extract underlying price, options data, and futures data.
        """
        self.cur_slice = slice
        # self.algorithm.log("collect_state")
        for etf in self.equity_symbols:
            macd_value = self.macd_map[etf].Current.Value
            macd_signal = self.macd_map[etf].Signal.Current.Value
            macd_histogram = self.macd_map[etf].Fast.Current.Value - self.macd_map[etf].Slow.Current.Value
            bollinger_upper = self.boll_map[etf].UpperBand.Current.Value
            bollinger_lower = self.boll_map[etf].LowerBand.Current.Value

            option_symbol = self.option_symbols[etf]
            future_symbol = self.future_symbols[self.sector_map[(etf, False)][0][0]]

            if option_symbol in slice.option_chains and future_symbol in slice.future_chains:
                options_chain = slice.option_chains[option_symbol]
                futures_chain = slice.future_chains[future_symbol]
                underlying_price = self.algorithm.securities[etf].price

                # Select the ATM call option
                atm_calls = [x for x in options_chain if x.right == OptionRight.CALL and abs(x.strike - underlying_price) <= 1]
                atm_puts = [x for x in options_chain if x.right == OptionRight.PUT and abs(x.strike - underlying_price) <= 1]
                if atm_calls and atm_puts:
                    self.contract_dict[etf] = (sorted(atm_calls, key=lambda x: x.expiry)[-1], sorted(atm_puts, key=lambda x: x.expiry)[-1])
                    self.call = self.contract_dict[etf][0]
                    self.put = self.contract_dict[etf][1]
                    # Build observation [underlying price, option price, delta, IV, time to expiry]
                    self.current_obs[etf] = np.array([
                        #Adding equity stuff
                        underlying_price,
                        macd_value,
                        macd_signal,
                        macd_histogram,
                        bollinger_upper,
                        bollinger_lower,

                        #Options stuff
                        self.call.last_price,
                        self.call.greeks.delta,
                        self.call.implied_volatility,
                        self.put.last_price,
                        self.put.greeks.delta,
                        self.put.implied_volatility,
                        (self.call.expiry - self.algorithm.time).days,
                        (self.put.expiry - self.algorithm.time).days
                    ], dtype=np.float32)

    def step_training(self, slice: Slice):
        """
        Take a step in the environment, simulate the RL decision process.
        Use the model to select the action.
        """
        # self.algorithm.log("ran step training")
        if self.current_obs is None:
            return

        # Use the model to predict the next action

        for etf in self.current_obs.keys():
            action, _states = self.algorithm.model.predict(self.current_obs[etf], deterministic=True)

        # Calculate reward
        reward = 0
        if self.contract and self.contract.symbol and self.contract.symbol in slice and slice[self.contract.symbol]:
            new_price = slice[self.contract.symbol].price
            if new_price:
                if self.prev_portfolio_val:
                    reward = self.algorithm.portfolio.total_portfolio_value - self.prev_portfolio_val  # Reward is based on price change
                else:
                    reward = -10

        # Execute action (this will be the trading logic based on action)
        done = False
        info = {}

        self.reward_history.append(reward)
        self.obs_history.append(self.current_obs[self.algorithm.equity_str])
        

        # Execute action based on the model prediction
        for etf, _ in self.algorithm.sector_map.keys():
            self._execute_action(action, etf, 1)

        return self.current_obs[self.algorithm.equity_str], reward, done, info

    def step(self, action):
        # self.algorithm.log("ran step training")
        if self.current_obs is None:
            return

        # Use the model to predict the next action
        for etf in self.current_obs.keys():
            action, _states = self.algorithm.model.predict(self.current_obs[etf], deterministic=True)

        # Calculate reward
        reward = float(0)
        # if self.contract and self.contract.symbol and self.contract.symbol in self.cur_slice and self.cur_slice[self.contract.symbol]:
        #     new_price = self.cur_slice[self.contract.symbol].price
        #     if new_price:
        if self.prev_portfolio_val:
            reward = self.algorithm.portfolio.total_portfolio_value - self.prev_portfolio_val  # Reward is based on price change
        else:
            reward = float(self.scale_factor)
            action = np.array(random.randint(1, 2))

        reward /= float(self.scale_factor)
        returns = np.array(self.reward_history[-30:])  # short window
        if len(returns) >= 2:
            mean = np.mean(returns)
            std = np.std(returns) + 1e-8
            sharpe = mean / std
            reward += float(sharpe)  # Encourage risk-adjusted return
        # # Execute action (this will be the trading logic based on action)

        if(reward <= 0 and action == 0): #and len(self.reward_history) > 0
            #if(self.reward_history[-1] == 0):
            self.algorithm.log("Holding while no reward penalty incurred")
            reward = -1
        # if(reward > 0 and action == 0):
        #     reward = 1
        if(action == 3 and not self.algorithm.portfolio.invested):
            self.algorithm.log("Selling while nothing in portfolio penalty incurred")
            reward = -1
        done = False
        info = {}
        self.algorithm.log(reward)
        self.reward_history.append(reward)
        self.obs_history.append(self.current_obs[self.algorithm.equity_str])
        self.prev_portfolio_val = self.algorithm.portfolio.total_portfolio_value

        # Execute action based on the model prediction
        self.algorithm.log(reward)
        for etf, _ in self.algorithm.sector_map.keys():
            self._execute_action(action, etf, 1)

        return self.current_obs[self.algorithm.equity_str], reward, done, info

    def run_live(self, slice: Slice):
        """
        Use a trained RL model to predict and execute live trades.
        """
        self.collect_state(slice)

        if self.current_obs is None:
            return

        # Get the action from the model for live trading
        for etf, _ in self.algorithm.sector_map.keys():
            action, _ = self.algorithm.model.predict(self.current_obs[etf], deterministic=True)
            self._execute_action(action, etf, 1)


    def _execute_action(self, action, etf: str, quantity):
        """
        Translate the action into a trade in QuantConnect for the given ETF.
        """
        self.algorithm.log(action)
        action = action.item()


        # Fetch the stored option contract for this ETF
        call_contract, put_contract = self.contract_dict[etf]
        option_symbol = self.option_symbols[etf]
        if not call_contract or not put_contract:
            return

        # Buy Call & Short Future
        future_name = self.sector_map[(etf, False)][0][0]
        future_symbol = self.future_symbols[future_name]
        # self.algorithm.log((option_symbol, future_symbol, option_symbol in self.algorithm.current_slice.option_chains, future_symbol in self.algorithm.current_slice.future_chains))
        
        # current_price = self.algorithm.securities[contract.symbol].price

        # # Initialize or update trailing stop
        # if contract.symbol not in self.trailing_stop_prices:
        #     self.trailing_stop_prices[contract.symbol] = current_price * (1 - self.stop_loss_pct)
        # else:
        #     # Update stop price only if price moved up
        #     self.trailing_stop_prices[contract.symbol] = max(
        #         self.trailing_stop_prices[contract.symbol],
        #         current_price * (1 - self.stop_loss_pct)
        #     )

        # # Check for stop-out
        # if self.algorithm.portfolio[contract.symbol].invested:
        #     if current_price <= self.trailing_stop_prices[contract.symbol]:
        #         self.algorithm.log(f"TRAILING STOP HIT for {contract.symbol}. Price: {current_price}, Stop: {self.trailing_stop_prices[contract.symbol]}")
        #         self.algorithm.liquidate(contract.symbol)
        #         if future_symbol in self.algorithm.current_slice.future_chains:
        #             future_chain = self.algorithm.current_slice.future_chains.get(future_symbol)
        #             if future_chain:
        #                 future_contract = sorted(future_chain, key=lambda x: x.expiry)[0]
        #                 self.algorithm.liquidate(future_contract.symbol)
        #         return

        if action == 1:
            if (future_symbol in self.algorithm.current_slice.future_chains and option_symbol in self.algorithm.current_slice.option_chains):
                self.algorithm.log("call + short fut")
                #if not self.algorithm.portfolio[contract.symbol].invested:
                self.algorithm.market_order(call_contract.symbol, 2 * quantity)
                self.algorithm.log(f"Contract: {call_contract.symbol} | Strike: {call_contract.strike} | Expiry: {call_contract.expiry} | Price: {call_contract.last_price} | Delta: {call_contract.greeks.delta} | Gamma: {call_contract.greeks.gamma}")

                
                future_chain = self.algorithm.current_slice.future_chains.get(future_symbol)
                if future_chain:
                    future_contract = sorted(future_chain, key=lambda x: x.expiry)[0]
                    self.algorithm.market_order(future_contract.symbol, -quantity)

        # Buy Put & Long Future
        elif action == 2:
            if (future_symbol in self.algorithm.current_slice.future_chains and option_symbol in self.algorithm.current_slice.option_chains):
                self.algorithm.log("put + long fut")
                #if not self.algorithm.portfolio[contract.symbol].invested:
                self.algorithm.market_order(put_contract.symbol, 2 * quantity)
                self.algorithm.log(f"Contract: {put_contract.symbol} | Strike: {put_contract.strike} | Expiry: {put_contract.expiry} | Price: {put_contract.last_price} | Delta: {put_contract.greeks.delta} | Gamma: {put_contract.greeks.gamma}")

                future_chain = self.algorithm.current_slice.future_chains.get(future_symbol)
                if future_chain:
                    future_contract = sorted(future_chain, key=lambda x: x.expiry)[0]
                    self.algorithm.market_order(future_contract.symbol, quantity)

        # elif action == 3:
        #     if (future_symbol in self.algorithm.current_slice.future_chains and option_symbol in self.algorithm.current_slice.option_chains):
        #         self.algorithm.log("sell")
        #         if self.algorithm.portfolio[option_symbol].invested:
        #             self.algorithm.liquidate(option_symbol)

        #         future_chain = self.algorithm.current_slice.future_chains.get(future_symbol)
        #         if future_chain:
        #             future_contract = sorted(future_chain, key=lambda x : x.expiry)[0]
        #             self.algorithm.liquidate(future_contract.symbol)
