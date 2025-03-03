import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dateutil.relativedelta import relativedelta
import time
import ta
import logging
import os
import ccxt

class ETHBacktester:
    def __init__(self, initial_capital=2000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.stablecoin_reserve = 0  # No stablecoin reserve needed for backtesting
        self.trading_capital = initial_capital  # Use full capital for trading
        self.position_size = 0
        self.in_position = False
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.trailing_stop = None
        self.trailing_stop_active = False
        
        # Settings - more aggressive
        self.risk_per_trade = 0.03          # 3% risk per trade (increased from 2%)
        self.stop_loss_percentage = 0.04    # 4% stop loss (reduced from 5%)
        self.take_profit_percentage = 0.12  # 12% take profit (reduced from 15%)
        self.trailing_stop_activation = 0.04  # Activate trailing stop after 4% profit (reduced from 5%)
        self.trailing_stop_distance = 0.025   # 2.5% trailing stop distance (reduced from 3%)
        
        # Performance tracking
        self.trades = []
        self.equity_curve = []
        self.drawdowns = []
        
        # Set up logging
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging to file and console"""
        # Create a unique log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"eth_backtest_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()  # Also output to console
            ]
        )
        
        self.logger = logging.getLogger("ETHBacktester")
        self.logger.info(f"Starting new backtest with ${self.initial_capital:.2f} initial capital")
        self.logger.info(f"Log file created: {log_filename}")
        
    def fetch_historical_data(self, days=365):
        """Fetch historical OHLCV data for ETH/USDT for a specific date range"""
        # Set explicit end date as February 3, 2025
        end_date = datetime(2025, 2, 3)
        # Calculate start date as 1 year before
        start_date = end_date - timedelta(days=days)
        
        print(f"Fetching historical ETH data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Try multiple exchanges in case one is restricted
        exchanges_to_try = ['kucoin', 'kraken', 'coinbase']
        
        # Fetch 1h, 4h and 1d timeframes
        timeframes = {'1h': '1h', '4h': '4h', '1d': '1d'}
        data = {}
        
        for exchange_id in exchanges_to_try:
            try:
                print(f"Trying to fetch data from {exchange_id}...")
                exchange = getattr(ccxt, exchange_id)({
                    'enableRateLimit': True,
                    'timeout': 30000,  # Increase timeout to 30 seconds
                })
                
                success = True
                for name, timeframe in timeframes.items():
                    try:
                        # For longer periods, we need to fetch data in chunks
                        all_ohlcv = []
                        since = int(start_date.timestamp() * 1000)
                        until = int(end_date.timestamp() * 1000)
                        
                        print(f"Fetching {timeframe} data from {exchange_id} for period: {start_date} to {end_date}")
                        
                        # Fetch data in chunks of 1000 candles
                        while since < until:
                            print(f"Fetching chunk from {datetime.fromtimestamp(since/1000)}")
                            ohlcv = exchange.fetch_ohlcv(
                                symbol='ETH/USDT',
                                timeframe=timeframe,
                                since=since,
                                limit=1000  # Maximum candles per request
                            )
                            
                            if not ohlcv or len(ohlcv) == 0:
                                print(f"No data returned for this chunk, stopping")
                                break
                                
                            all_ohlcv.extend(ohlcv)
                            
                            # Update since for next iteration - use the timestamp of the last candle + 1
                            last_timestamp = ohlcv[-1][0]
                            since = last_timestamp + 1
                            
                            print(f"Fetched {len(ohlcv)} candles, last timestamp: {datetime.fromtimestamp(last_timestamp/1000)}")
                            
                            # Rate limiting to avoid API restrictions
                            time.sleep(exchange.rateLimit / 1000)
                            
                            # Break if we've reached end date
                            if last_timestamp >= until:
                                break
                        
                        if not all_ohlcv:
                            print(f"No data fetched for {timeframe} from {exchange_id}")
                            success = False
                            break
                            
                        # Convert to DataFrame
                        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        
                        # Filter to ensure we only have data within our date range
                        df = df[(df.index >= start_date) & (df.index <= end_date)]
                        
                        # Remove duplicates if any
                        df = df[~df.index.duplicated(keep='first')]
                        
                        # Sort by timestamp to ensure chronological order
                        df = df.sort_index()
                        
                        # Verify data range
                        print(f"Data range for {timeframe}: {df.index[0]} to {df.index[-1]}, {len(df)} candles")
                        
                        # Calculate indicators
                        df = self.calculate_indicators(df)
                        
                        data[name] = df
                        
                    except Exception as e:
                        print(f"Error fetching {timeframe} data from {exchange_id}: {str(e)}")
                        success = False
                        break
                
                if success and all(tf in data for tf in timeframes):
                    print(f"Successfully fetched all data from {exchange_id}")
                    return data
                    
            except Exception as e:
                print(f"Error initializing {exchange_id}: {str(e)}")
        
        # If we couldn't get data from any exchange, raise an error
        if not data:
            raise Exception("Could not fetch data from any exchange. Please check your internet connection and exchange availability.")
        
        return data
    
    def calculate_indicators(self, df):
        """Calculate technical indicators for analysis"""
        if len(df) < 50:
            return df
        
        # Copy the dataframe to avoid warnings
        df = df.copy()
        
        # Add EMAs
        df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['ema21'] = ta.trend.ema_indicator(df['close'], window=21)
        df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
        df['ema200'] = ta.trend.ema_indicator(df['close'], window=200)
        
        # Add SMAs
        df['sma20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['sma200'] = ta.trend.sma_indicator(df['close'], window=200)
        
        # Add RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # Add MACD
        macd = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # Add Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        
        # Add ATR for volatility
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        
        # Add ADX for trend strength
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()
        
        return df
    
    def analyze_markets(self, data, timestamp):
        """Analyze market data for a specific timestamp across timeframes"""
        analysis_results = {}
        
        for timeframe, df in data.items():
            # Get data up to the timestamp (to avoid lookahead bias)
            history = df[df.index <= timestamp].copy()
            
            if len(history) < 50:  # Need enough data for indicators
                self.logger.warning(f"Not enough data for {timeframe} at {timestamp}: {len(history)} points")
                continue
                
            analysis = {}
            
            # Get the latest and previous data points
            latest = history.iloc[-1]
            prev = history.iloc[-2]
            
            # Log data types and shapes for debugging
            self.logger.debug(f"Analyzing {timeframe} at {timestamp}")
            self.logger.debug(f"latest['volume'] type: {type(latest['volume'])}, value: {latest['volume']}")
            
            # === IMPROVED STRATEGY LOGIC ===
            
            # 1. Trend strength using ADX
            adx_threshold = 25
            analysis['trend_strength'] = 'strong' if latest['adx'] > adx_threshold else 'weak'
            
            # 2. Enhanced trend direction using multiple EMAs
            ema_short = latest['ema9']
            ema_medium = latest['ema21']
            ema_long = latest['ema50']
            
            # Check if EMAs are aligned for strong trend
            if ema_short > ema_medium > ema_long:
                analysis['trend'] = 'strong_bullish'
            elif ema_short < ema_medium < ema_long:
                analysis['trend'] = 'strong_bearish'
            elif ema_short > ema_medium:
                analysis['trend'] = 'bullish'
            elif ema_short < ema_medium:
                analysis['trend'] = 'bearish'
            else:
                analysis['trend'] = 'neutral'
            
            # 3. Volatility assessment using ATR
            atr_percent = latest['atr'] / latest['close'] * 100
            if atr_percent > 3:
                analysis['volatility'] = 'high'
            elif atr_percent > 1.5:
                analysis['volatility'] = 'medium'
            else:
                analysis['volatility'] = 'low'
            
            # 4. Momentum with RSI and MACD
            analysis['momentum'] = 'neutral'
            if latest['rsi'] < 30:
                analysis['momentum'] = 'oversold'
            elif latest['rsi'] > 70:
                analysis['momentum'] = 'overbought'
            
            # 5. MACD signal with histogram direction
            analysis['macd_signal'] = 'neutral'
            macd_hist = latest['macd'] - latest['macd_signal']
            prev_macd_hist = prev['macd'] - prev['macd_signal']
            
            if latest['macd'] > latest['macd_signal']:
                if macd_hist > prev_macd_hist:
                    analysis['macd_signal'] = 'strong_bullish'
                else:
                    analysis['macd_signal'] = 'bullish'
            elif latest['macd'] < latest['macd_signal']:
                if macd_hist < prev_macd_hist:
                    analysis['macd_signal'] = 'strong_bearish'
                else:
                    analysis['macd_signal'] = 'bearish'
            
            # 6. Support/Resistance with Bollinger Bands
            analysis['bb_signal'] = 'neutral'
            bb_width = (latest['bb_upper'] - latest['bb_lower']) / latest['bb_middle']
            
            if latest['close'] < latest['bb_lower']:
                analysis['bb_signal'] = 'oversold'
            elif latest['close'] > latest['bb_upper']:
                analysis['bb_signal'] = 'overbought'
            
            # 7. Volume analysis - FIX THE COMPARISON ISSUE
            analysis['volume_signal'] = 'neutral'
            
            # Calculate volume moving average
            volume_ma = history['volume'].rolling(20).mean().iloc[-1]
            
            # Log the values for debugging
            self.logger.debug(f"Volume: {latest['volume']}, Volume MA: {volume_ma}, Threshold: {volume_ma * 1.5}")
            
            # Compare scalar values properly
            if latest['volume'] > (volume_ma * 1.5):
                if latest['close'] > latest['open']:
                    analysis['volume_signal'] = 'bullish'
                else:
                    analysis['volume_signal'] = 'bearish'
            
            # === ENTRY/EXIT SIGNAL CALCULATION ===
            
            # Calculate entry signals with weighted approach
            analysis['entry_signals'] = 0
            
            # Strong trend conditions for entry
            if analysis['trend'] == 'strong_bullish' and analysis['trend_strength'] == 'strong':
                analysis['entry_signals'] += 2.0
            elif analysis['trend'] == 'bullish':
                analysis['entry_signals'] += 1.0
            
            # Momentum conditions
            if analysis['momentum'] == 'oversold':
                analysis['entry_signals'] += 1.5
            
            # MACD conditions
            if analysis['macd_signal'] == 'strong_bullish':
                analysis['entry_signals'] += 1.5
            elif analysis['macd_signal'] == 'bullish':
                analysis['entry_signals'] += 0.75
            
            # BB conditions
            if analysis['bb_signal'] == 'oversold':
                analysis['entry_signals'] += 1.0
            
            # Volume confirmation
            if analysis['volume_signal'] == 'bullish':
                analysis['entry_signals'] += 1.0
            
            # Calculate exit signals
            analysis['exit_signals'] = 0
            
            # Strong trend conditions for exit
            if analysis['trend'] == 'strong_bearish' and analysis['trend_strength'] == 'strong':
                analysis['exit_signals'] += 2.0
            elif analysis['trend'] == 'bearish':
                analysis['exit_signals'] += 1.0
            
            # Momentum conditions
            if analysis['momentum'] == 'overbought':
                analysis['exit_signals'] += 1.5
            
            # MACD conditions
            if analysis['macd_signal'] == 'strong_bearish':
                analysis['exit_signals'] += 1.5
            elif analysis['macd_signal'] == 'bearish':
                analysis['exit_signals'] += 0.75
            
            # BB conditions
            if analysis['bb_signal'] == 'overbought':
                analysis['exit_signals'] += 1.0
            
            # Volume confirmation
            if analysis['volume_signal'] == 'bearish':
                analysis['exit_signals'] += 1.0
            
            analysis_results[timeframe] = analysis
        
        return analysis_results
    
    def make_decision(self, analysis_results, current_price, timestamp):
        """Make trading decision based on analysis results"""
        if not analysis_results:  # Skip if no analysis available
            return 'wait'
            
        # Combine signals from different timeframes with more weight to shorter timeframes
        timeframe_weights = {'1h': 0.5, '4h': 0.3, '1d': 0.2}
        entry_signals = 0
        exit_signals = 0
        
        for tf in analysis_results:
            weight = timeframe_weights.get(tf, 0.33)
            entry_signals += analysis_results[tf]['entry_signals'] * weight
            exit_signals += analysis_results[tf]['exit_signals'] * weight
        
        # Calculate signal strength (0-1)
        max_possible_signals = sum(timeframe_weights.values()) * 7  # 7 signals per timeframe
        entry_strength = entry_signals / max_possible_signals if max_possible_signals > 0 else 0
        exit_strength = exit_signals / max_possible_signals if max_possible_signals > 0 else 0
        
        # Lower thresholds for more trades
        entry_threshold = 0.35  # Lowered from 0.5
        exit_threshold = 0.35   # Lowered from 0.4
        
        # Make decision
        decision = 'wait'
        if self.in_position:
            # For existing positions, consider exit
            if exit_strength > exit_threshold:
                decision = 'sell'
                self.logger.info(f"\n!!! SELL SIGNAL at {timestamp} !!!")
                self.logger.info(f"Price: ${current_price:.2f}")
                self.logger.info(f"Exit strength: {exit_strength:.2f} (threshold: {exit_threshold})")
                
                # Log detailed exit reasons
                self.logger.info("Exit signal details:")
                for tf in analysis_results:
                    analysis = analysis_results[tf]
                    self.logger.info(f"  {tf}: {analysis['exit_signals']} signals")
                    if analysis['trend'] in ['bearish', 'strong_bearish']:
                        self.logger.info(f"    - {tf} trend: {analysis['trend']}")
                    if analysis['momentum'] == 'overbought':
                        self.logger.info(f"    - {tf} RSI: {analysis['momentum']}")
                    if analysis['macd_signal'] in ['bearish', 'strong_bearish']:
                        self.logger.info(f"    - {tf} MACD: {analysis['macd_signal']}")
                    if analysis['bb_signal'] == 'overbought':
                        self.logger.info(f"    - {tf} BB: {analysis['bb_signal']}")
            else:
                decision = 'hold'
        else:
            # For potential new positions
            if entry_strength > entry_threshold:
                # Check if we have available trading capital
                if self.trading_capital > 0:
                    decision = 'buy'
                    self.logger.info(f"\n!!! BUY SIGNAL at {timestamp} !!!")
                    self.logger.info(f"Price: ${current_price:.2f}")
                    self.logger.info(f"Entry strength: {entry_strength:.2f} (threshold: {entry_threshold})")
                    
                    # Log detailed entry reasons
                    self.logger.info("Entry signal details:")
                    for tf in analysis_results:
                        analysis = analysis_results[tf]
                        self.logger.info(f"  {tf}: {analysis['entry_signals']} signals")
                        if analysis['trend'] in ['bullish', 'strong_bullish']:
                            self.logger.info(f"    - {tf} trend: {analysis['trend']}")
                        if analysis['momentum'] == 'oversold':
                            self.logger.info(f"    - {tf} RSI: {analysis['momentum']}")
                        if analysis['macd_signal'] in ['bullish', 'strong_bullish']:
                            self.logger.info(f"    - {tf} MACD: {analysis['macd_signal']}")
                        if analysis['bb_signal'] == 'oversold':
                            self.logger.info(f"    - {tf} BB: {analysis['bb_signal']}")
        
        return decision
    
    def execute_trade(self, action, price, timestamp):
        """Execute a trade based on the decision"""
        if action == 'buy' and not self.in_position:
            # Calculate position size based on risk
            risk_amount = self.trading_capital * self.risk_per_trade
            stop_loss_price = price * (1 - self.stop_loss_percentage)
            risk_per_unit = price - stop_loss_price
            position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
            
            # Calculate cost in USD
            cost = position_size * price
            
            # Check if we have enough funds
            if cost <= self.trading_capital:
                # Execute buy order
                self.trading_capital -= cost
                self.position_size = position_size
                self.entry_price = price
                self.in_position = True
                self.stop_loss = stop_loss_price
                self.take_profit = price * (1 + self.take_profit_percentage)
                self.trailing_stop_active = False
                self.trailing_stop = None
                
                # Record the trade
                self.trades.append({
                    'action': 'buy',
                    'timestamp': timestamp,
                    'price': price,
                    'position_size': position_size,
                    'cost': cost,
                    'stop_loss': self.stop_loss,
                    'take_profit': self.take_profit
                })
                
                self.logger.info(f"BUY at {timestamp}: {position_size:.6f} ETH at ${price:.2f}, " + 
                      f"Cost: ${cost:.2f}, Stop: ${self.stop_loss:.2f}, Target: ${self.take_profit:.2f}")
        
        elif action == 'sell' and self.in_position:
            # Calculate profit/loss
            pl_percent = (price - self.entry_price) / self.entry_price
            pl_amount = self.position_size * (price - self.entry_price)
            
            # Execute sell order
            sale_value = self.position_size * price
            self.trading_capital += sale_value
            
            # Record the trade
            self.trades.append({
                'action': 'sell',
                'timestamp': timestamp,
                'price': price,
                'position_size': self.position_size,
                'sale_value': sale_value,
                'pl_percent': pl_percent,
                'pl_amount': pl_amount
            })
            
            self.logger.info(f"SELL at {timestamp}: {self.position_size:.6f} ETH at ${price:.2f}, " + 
                  f"P/L: {pl_percent:.2%} (${pl_amount:.2f})")
            
            # Reset position
            self.position_size = 0
            self.in_position = False
            self.entry_price = 0
            self.stop_loss = 0
            self.take_profit = 0
            self.trailing_stop_active = False
            self.trailing_stop = None
    
    def check_stop_loss(self, price, timestamp):
        """Check if stop loss or take profit has been triggered"""
        if not self.in_position:
            return False
        
        # Check if price hit take profit
        if price >= self.take_profit:
            self.logger.info(f"Take profit hit at {timestamp}: ${price:.2f}")
            self.execute_trade('sell', price, timestamp)
            return True
        
        # Check if trailing stop should be activated
        profit_percent = (price - self.entry_price) / self.entry_price
        if (not self.trailing_stop_active and 
            profit_percent >= self.trailing_stop_activation):
            self.trailing_stop_active = True
            self.trailing_stop = price * (1 - self.trailing_stop_distance)
            self.logger.info(f"Trailing stop activated at {timestamp}: ${self.trailing_stop:.2f}")
        
        # Update trailing stop if needed
        if self.trailing_stop_active:
            new_stop = price * (1 - self.trailing_stop_distance)
            if new_stop > self.trailing_stop:
                self.trailing_stop = new_stop
                self.logger.debug(f"Trailing stop updated at {timestamp}: ${self.trailing_stop:.2f}")
        
        # Check if price hit stop loss or trailing stop
        stop_price = self.trailing_stop if self.trailing_stop_active else self.stop_loss
        if price <= stop_price:
            self.logger.info(f"Stop loss hit at {timestamp}: ${price:.2f}")
            self.execute_trade('sell', price, timestamp)
            return True
        
        return False
    
    def calculate_portfolio_value(self, price):
        """Calculate total portfolio value"""
        portfolio_value = self.stablecoin_reserve + self.trading_capital
        if self.in_position:
            portfolio_value += self.position_size * price
        return portfolio_value
    
    def run_backtest(self, market_data):
        """Run backtest on historical data"""
        self.logger.info(f"Starting backtest with ${self.initial_capital:.2f} initial capital...")
        
        # Get hourly data for detailed price movements
        hourly_data = market_data['1h']
        
        # Log data summary
        self.logger.info(f"Data summary:")
        for timeframe, df in market_data.items():
            self.logger.info(f"  {timeframe}: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        
        # Track equity curve
        equity_values = []
        timestamps = []
        max_equity = self.initial_capital
        current_drawdown = 0
        max_drawdown = 0
        
        # Track signal statistics
        entry_strengths = []
        exit_strengths = []
        
        # Iterate through each hour
        for idx, row in hourly_data.iterrows():
            timestamp = idx
            current_price = row['close']
            
            # First check if stop loss or take profit is hit
            if self.in_position:
                stop_triggered = self.check_stop_loss(current_price, timestamp)
                if stop_triggered:
                    # Skip further analysis if we just exited a position
                    portfolio_value = self.calculate_portfolio_value(current_price)
                    equity_values.append(portfolio_value)
                    timestamps.append(timestamp)
                    continue
            
            # Analyze market data across timeframes
            analysis_results = self.analyze_markets(market_data, timestamp)
            
            # Make trading decision
            decision = self.make_decision(analysis_results, current_price, timestamp)
            
            # Track signal strengths for analysis
            if analysis_results:
                # Calculate signal strengths
                timeframe_weights = {'1h': 0.5, '4h': 0.3, '1d': 0.2}
                entry_signal = sum([analysis_results[tf]['entry_signals'] * timeframe_weights.get(tf, 0.33) for tf in analysis_results])
                exit_signal = sum([analysis_results[tf]['exit_signals'] * timeframe_weights.get(tf, 0.33) for tf in analysis_results])
                
                max_possible = sum(timeframe_weights.values()) * 7
                entry_strength = entry_signal / max_possible if max_possible > 0 else 0
                exit_strength = exit_signal / max_possible if max_possible > 0 else 0
                
                entry_strengths.append(entry_strength)
                exit_strengths.append(exit_strength)
                
                # Log detailed analysis periodically (daily at midnight)
                if timestamp.hour == 0 and timestamp.minute == 0:
                    self.logger.info(f"\n=== Daily Analysis at {timestamp} ===")
                    self.logger.info(f"Price: ${current_price:.2f}")
                    self.logger.info(f"Portfolio value: ${self.calculate_portfolio_value(current_price):.2f}")
                    self.logger.info(f"Entry strength: {entry_strength:.4f}, Exit strength: {exit_strength:.4f}")
                    
                    for tf in analysis_results:
                        analysis = analysis_results[tf]
                        self.logger.info(f"\n{tf} Analysis:")
                        for key, value in analysis.items():
                            if key not in ['entry_signals', 'exit_signals']:
                                self.logger.info(f"  {key}: {value}")
                        self.logger.info(f"  Entry signals: {analysis['entry_signals']}")
                        self.logger.info(f"  Exit signals: {analysis['exit_signals']}")
            
            # Execute trade if needed
            if decision in ['buy', 'sell']:
                self.execute_trade(decision, current_price, timestamp)
            
            # Track equity curve
            portfolio_value = self.calculate_portfolio_value(current_price)
            equity_values.append(portfolio_value)
            timestamps.append(timestamp)
            
            # Track drawdown
            if portfolio_value > max_equity:
                max_equity = portfolio_value
                current_drawdown = 0
            else:
                current_drawdown = (max_equity - portfolio_value) / max_equity
                if current_drawdown > max_drawdown:
                    max_drawdown = current_drawdown
                    self.logger.info(f"New maximum drawdown: {max_drawdown:.2%} at {timestamp}")
        
        # Store final equity curve and drawdown
        self.equity_curve = list(zip(timestamps, equity_values))
        self.max_drawdown = max_drawdown
        
        # Log signal strength statistics
        if entry_strengths:
            self.logger.info("\nSignal Strength Statistics:")
            self.logger.info(f"Entry strength - Min: {min(entry_strengths):.4f}, Max: {max(entry_strengths):.4f}, Avg: {sum(entry_strengths)/len(entry_strengths):.4f}")
            self.logger.info(f"Exit strength - Min: {min(exit_strengths):.4f}, Max: {max(exit_strengths):.4f}, Avg: {sum(exit_strengths)/len(exit_strengths):.4f}")
            
            # Count how many times signals were close to thresholds
            entry_threshold = 0.35
            exit_threshold = 0.35
            close_entries = sum(1 for s in entry_strengths if s > entry_threshold * 0.8 and s < entry_threshold)
            close_exits = sum(1 for s in exit_strengths if s > exit_threshold * 0.8 and s < exit_threshold)
            
            self.logger.info(f"Signals close to but below entry threshold: {close_entries}")
            self.logger.info(f"Signals close to but below exit threshold: {close_exits}")
        
        # Calculate performance metrics
        final_value = equity_values[-1] if equity_values else self.initial_capital
        total_return = (final_value / self.initial_capital - 1) * 100
        
        # Calculate win rate and average trade
        winning_trades = [t for t in self.trades if t['action'] == 'sell' and t.get('pl_amount', 0) > 0]
        losing_trades = [t for t in self.trades if t['action'] == 'sell' and t.get('pl_amount', 0) <= 0]
        
        total_trades = len(winning_trades) + len(losing_trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Calculate average profit and loss
        avg_profit = sum([t.get('pl_amount', 0) for t in winning_trades]) / len(winning_trades) if winning_trades else 0
        avg_loss = sum([t.get('pl_amount', 0) for t in losing_trades]) / len(losing_trades) if losing_trades else 0
        
        # Calculate profit factor
        gross_profit = sum([t.get('pl_amount', 0) for t in winning_trades])
        gross_loss = abs(sum([t.get('pl_amount', 0) for t in losing_trades]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # If still in position, calculate unrealized P&L
        unrealized_pl = 0
        if self.in_position:
            last_price = hourly_data.iloc[-1]['close']
            unrealized_pl = self.position_size * (last_price - self.entry_price)
        
        # Print results
        self.logger.info("\n===== BACKTEST RESULTS =====")
        self.logger.info(f"Testing Period: {hourly_data.index[0]} to {hourly_data.index[-1]}")
        self.logger.info(f"Initial Capital: ${self.initial_capital:.2f}")
        self.logger.info(f"Final Capital: ${final_value:.2f}")
        self.logger.info(f"Total Return: {total_return:.2f}%")
        self.logger.info(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")
        self.logger.info(f"Total Trades: {total_trades}")
        self.logger.info(f"Win Rate: {win_rate * 100:.2f}%")
        self.logger.info(f"Average Profit: ${avg_profit:.2f}")
        self.logger.info(f"Average Loss: ${avg_loss:.2f}")
        self.logger.info(f"Profit Factor: {profit_factor:.2f}")
        if self.in_position:
            self.logger.info(f"Currently In Position: {self.position_size:.6f} ETH @ ${self.entry_price:.2f}")
            self.logger.info(f"Unrealized P&L: ${unrealized_pl:.2f}")
        
        # Log all trades
        self.logger.info("\n===== TRADE HISTORY =====")
        for i, trade in enumerate(self.trades):
            if trade['action'] == 'buy':
                self.logger.info(f"Trade {i+1}: BUY {trade['position_size']:.6f} ETH at ${trade['price']:.2f} on {trade['timestamp']}")
            elif trade['action'] == 'sell':
                pl_percent = trade.get('pl_percent', 0) * 100
                pl_amount = trade.get('pl_amount', 0)
                self.logger.info(f"Trade {i+1}: SELL {trade['position_size']:.6f} ETH at ${trade['price']:.2f} on {trade['timestamp']} - P/L: {pl_percent:.2f}% (${pl_amount:.2f})")
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': final_value,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'equity_curve': self.equity_curve,
            'trades': self.trades,
            'entry_strengths': entry_strengths,
            'exit_strengths': exit_strengths
        }
    
    def plot_results(self, results):
        """Plot backtest results"""
        print("Generating performance charts...")
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot equity curve
        timestamps = [t for t, v in results['equity_curve']]
        equity_values = [v for t, v in results['equity_curve']]
        
        ax1.plot(timestamps, equity_values, label='Portfolio Value')
        ax1.set_title('Backtest Results: ETH/USDT Last 30 Days')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True)
        ax1.legend()
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        # Mark buy and sell points
        buy_times = [t['timestamp'] for t in results['trades'] if t['action'] == 'buy']
        buy_values = [equity_values[timestamps.index(t)] for t in buy_times if t in timestamps]
        
        sell_times = [t['timestamp'] for t in results['trades'] if t['action'] == 'sell']
        sell_values = [equity_values[timestamps.index(t)] for t in sell_times if t in timestamps]
        
        ax1.scatter(buy_times, buy_values, color='green', marker='^', label='Buy', zorder=5)
        ax1.scatter(sell_times, sell_values, color='red', marker='v', label='Sell', zorder=5)
        ax1.legend()
        
        # Plot trade P&L
        trade_results = [t.get('pl_amount', 0) for t in results['trades'] if t['action'] == 'sell']
        trade_times = [t['timestamp'] for t in results['trades'] if t['action'] == 'sell']
        
        colors = ['green' if pl > 0 else 'red' for pl in trade_results]
        
        ax2.bar(trade_times, trade_results, color=colors, width=0.5)
        ax2.set_title('Trade P&L')
        ax2.set_ylabel('Profit/Loss ($)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('eth_backtest_results.png')
        print("Chart saved as 'eth_backtest_results.png'")
        
        # Display summary in the console
        print("\n===== TRADING SUMMARY =====")
        profitable_days = sum(1 for pl in trade_results if pl > 0)
        losing_days = sum(1 for pl in trade_results if pl <= 0)
        
        print(f"Profitable Trades: {profitable_days}")
        print(f"Losing Trades: {losing_days}")
        print(f"Largest Win: ${max(trade_results) if trade_results else 0:.2f}")
        print(f"Largest Loss: ${min(trade_results) if trade_results else 0:.2f}")
        print(f"Average Trade: ${sum(trade_results)/len(trade_results) if trade_results else 0:.2f}")
        
        return fig

# Run the backtest
if __name__ == "__main__":
    # Create backtester instance
    backtester = ETHBacktester(initial_capital=2000)
    
    # Fetch historical data for 1 year
    days = 365
    print(f"Starting backtest for the last {days} days")
    market_data = backtester.fetch_historical_data(days=days)
    
    # Run backtest
    results = backtester.run_backtest(market_data)
    
    # Plot results
    backtester.plot_results(results)