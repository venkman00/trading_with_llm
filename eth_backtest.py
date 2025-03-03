import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dateutil.relativedelta import relativedelta
import time

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
        """Calculate technical indicators on the dataframe"""
        # Calculate Moving Averages
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['sma50'] = df['close'].rolling(window=50).mean()
        df['sma200'] = df['close'].rolling(window=200).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['signal']
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        return df
    
    def analyze_markets(self, data, timestamp):
        """Analyze market data for a specific timestamp across timeframes"""
        analysis_results = {}
        
        for timeframe, df in data.items():
            # Get data up to the timestamp (to avoid lookahead bias)
            history = df[df.index <= timestamp].copy()
            
            if len(history) < 50:  # Need enough data for indicators
                continue
                
            analysis = {}
            
            # Get the latest and previous data points
            latest = history.iloc[-1]
            prev = history.iloc[-2]
            
            # Trend analysis - more sensitive to short-term trends
            analysis['trend'] = 'neutral'
            if latest['sma20'] > latest['sma50']:
                analysis['trend'] = 'bullish'
            elif latest['sma20'] < latest['sma50']:
                analysis['trend'] = 'bearish'
            
            # Momentum analysis - adjusted RSI thresholds for more trades
            analysis['momentum'] = 'neutral'
            if latest['rsi'] > 65:  # Lowered from 70
                analysis['momentum'] = 'overbought'
            elif latest['rsi'] < 35:  # Raised from 30
                analysis['momentum'] = 'oversold'
            
            # MACD analysis - more sensitive to crossovers
            analysis['macd_signal'] = 'neutral'
            if latest['macd'] > latest['signal'] and prev['macd'] <= prev['signal']:
                analysis['macd_signal'] = 'bullish_crossover'
            elif latest['macd'] < latest['signal'] and prev['macd'] >= prev['signal']:
                analysis['macd_signal'] = 'bearish_crossover'
            # Add MACD direction
            if latest['macd'] > prev['macd']:
                analysis['macd_direction'] = 'rising'
            else:
                analysis['macd_direction'] = 'falling'
            
            # BB analysis - more aggressive entries and exits
            analysis['bb_signal'] = 'neutral'
            if latest['close'] > latest['bb_upper'] * 0.98:  # 98% of upper band
                analysis['bb_signal'] = 'overbought'
            elif latest['close'] < latest['bb_lower'] * 1.02:  # 102% of lower band
                analysis['bb_signal'] = 'oversold'
            
            # Add price action signals
            analysis['price_action'] = 'neutral'
            # Bullish engulfing
            if prev['close'] < prev['open'] and latest['close'] > latest['open'] and latest['close'] > prev['open'] and latest['open'] < prev['close']:
                analysis['price_action'] = 'bullish'
            # Bearish engulfing
            elif prev['close'] > prev['open'] and latest['close'] < latest['open'] and latest['close'] < prev['open'] and latest['open'] > prev['close']:
                analysis['price_action'] = 'bearish'
            
            # Combine signals
            analysis['entry_signals'] = 0
            analysis['exit_signals'] = 0
            
            # Count entry signals - more weight to short-term signals
            if analysis['trend'] == 'bullish':
                analysis['entry_signals'] += 1
            if analysis['momentum'] == 'oversold':
                analysis['entry_signals'] += 1.5  # More weight to RSI
            if analysis['macd_signal'] == 'bullish_crossover':
                analysis['entry_signals'] += 1.5  # More weight to MACD crossover
            if analysis['macd_direction'] == 'rising':
                analysis['entry_signals'] += 0.5  # Add MACD direction
            if analysis['bb_signal'] == 'oversold':
                analysis['entry_signals'] += 1
            if analysis['price_action'] == 'bullish':
                analysis['entry_signals'] += 1
            
            # Count exit signals
            if analysis['trend'] == 'bearish':
                analysis['exit_signals'] += 1
            if analysis['momentum'] == 'overbought':
                analysis['exit_signals'] += 1.5  # More weight to RSI
            if analysis['macd_signal'] == 'bearish_crossover':
                analysis['exit_signals'] += 1.5  # More weight to MACD crossover
            if analysis['macd_direction'] == 'falling':
                analysis['exit_signals'] += 0.5  # Add MACD direction
            if analysis['bb_signal'] == 'overbought':
                analysis['exit_signals'] += 1
            if analysis['price_action'] == 'bearish':
                analysis['exit_signals'] += 1
            
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
        max_possible_signals = sum(timeframe_weights.values()) * 6  # 6 signals per timeframe
        entry_strength = entry_signals / max_possible_signals if max_possible_signals > 0 else 0
        exit_strength = exit_signals / max_possible_signals if max_possible_signals > 0 else 0
        
        # Log signal strengths periodically (every day)
        if timestamp.hour == 0 and timestamp.minute == 0:
            print(f"\n=== Daily Signal Analysis at {timestamp} ===")
            print(f"Price: ${current_price:.2f}")
            print(f"Entry signals: {entry_signals:.2f} / {max_possible_signals} (strength: {entry_strength:.2f})")
            print(f"Exit signals: {exit_signals:.2f} / {max_possible_signals} (strength: {exit_strength:.2f})")
            print(f"Current position: {'In position' if self.in_position else 'No position'}")
            print(f"Trading capital: ${self.trading_capital:.2f}")
            
            # Log detailed analysis for each timeframe
            for tf in analysis_results:
                analysis = analysis_results[tf]
                print(f"\n{tf} Analysis:")
                print(f"  Trend: {analysis['trend']}")
                print(f"  Momentum: {analysis['momentum']}")
                print(f"  MACD: {analysis['macd_signal']}")
                print(f"  BB: {analysis['bb_signal']}")
                if 'price_action' in analysis:
                    print(f"  Price Action: {analysis['price_action']}")
                print(f"  Entry signals: {analysis['entry_signals']}")
                print(f"  Exit signals: {analysis['exit_signals']}")
        
        # Log when we're close to a trading decision (within 80% of threshold)
        threshold_proximity = 0.8
        entry_threshold = 0.5
        exit_threshold = 0.4
        
        if entry_strength > entry_threshold * threshold_proximity and not self.in_position:
            print(f"\nClose to BUY at {timestamp}:")
            print(f"Price: ${current_price:.2f}")
            print(f"Entry strength: {entry_strength:.2f} (threshold: {entry_threshold})")
            for tf in analysis_results:
                print(f"  {tf} signals: {analysis_results[tf]['entry_signals']}")
        
        if exit_strength > exit_threshold * threshold_proximity and self.in_position:
            print(f"\nClose to SELL at {timestamp}:")
            print(f"Price: ${current_price:.2f}")
            print(f"Exit strength: {exit_strength:.2f} (threshold: {exit_threshold})")
            for tf in analysis_results:
                print(f"  {tf} signals: {analysis_results[tf]['exit_signals']}")
        
        # Log actual trading decisions
        decision = 'wait'
        if self.in_position:
            # For existing positions, consider exit
            if exit_strength > exit_threshold:
                decision = 'sell'
                print(f"\n!!! SELL SIGNAL at {timestamp} !!!")
                print(f"Price: ${current_price:.2f}")
                print(f"Exit strength: {exit_strength:.2f} (threshold: {exit_threshold})")
            else:
                decision = 'hold'
        else:
            # For potential new positions
            if entry_strength > entry_threshold:
                # Check if we have available trading capital
                if self.trading_capital > 0:
                    decision = 'buy'
                    print(f"\n!!! BUY SIGNAL at {timestamp} !!!")
                    print(f"Price: ${current_price:.2f}")
                    print(f"Entry strength: {entry_strength:.2f} (threshold: {entry_threshold})")
        
        # Log signal distribution histogram monthly
        if timestamp.day == 1 and timestamp.hour == 0 and timestamp.minute == 0:
            print(f"\n=== Monthly Signal Distribution as of {timestamp} ===")
            print(f"Entry strength distribution (threshold: {entry_threshold}):")
            print(f"Exit strength distribution (threshold: {exit_threshold}):")
        
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
                
                print(f"BUY at {timestamp}: {position_size:.6f} ETH at ${price:.2f}, " + 
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
            
            print(f"SELL at {timestamp}: {self.position_size:.6f} ETH at ${price:.2f}, " + 
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
            return
        
        # Check if price hit take profit
        if price >= self.take_profit:
            print(f"Take profit hit at {timestamp}: ${price:.2f}")
            self.execute_trade('sell', price, timestamp)
            return True
        
        # Check if trailing stop should be activated
        profit_percent = (price - self.entry_price) / self.entry_price
        if (not self.trailing_stop_active and 
            profit_percent >= self.trailing_stop_activation):
            self.trailing_stop_active = True
            self.trailing_stop = price * (1 - self.trailing_stop_distance)
            print(f"Trailing stop activated at {timestamp}: ${self.trailing_stop:.2f}")
        
        # Update trailing stop if needed
        if self.trailing_stop_active:
            new_stop = price * (1 - self.trailing_stop_distance)
            if new_stop > self.trailing_stop:
                self.trailing_stop = new_stop
        
        # Check if price hit stop loss or trailing stop
        stop_price = self.trailing_stop if self.trailing_stop_active else self.stop_loss
        if price <= stop_price:
            print(f"Stop loss hit at {timestamp}: ${price:.2f}")
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
        print(f"Starting backtest with ${self.initial_capital:.2f} initial capital...")
        
        # Get hourly data for detailed price movements
        hourly_data = market_data['1h']
        
        # Log data summary
        print(f"Data summary:")
        for timeframe, df in market_data.items():
            print(f"  {timeframe}: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        
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
                
                max_possible = sum(timeframe_weights.values()) * 6
                entry_strength = entry_signal / max_possible if max_possible > 0 else 0
                exit_strength = exit_signal / max_possible if max_possible > 0 else 0
                
                entry_strengths.append(entry_strength)
                exit_strengths.append(exit_strength)
            
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
        
        # Store final equity curve and drawdown
        self.equity_curve = list(zip(timestamps, equity_values))
        self.max_drawdown = max_drawdown
        
        # Log signal strength statistics
        if entry_strengths:
            print("\nSignal Strength Statistics:")
            print(f"Entry strength - Min: {min(entry_strengths):.4f}, Max: {max(entry_strengths):.4f}, Avg: {sum(entry_strengths)/len(entry_strengths):.4f}")
            print(f"Exit strength - Min: {min(exit_strengths):.4f}, Max: {max(exit_strengths):.4f}, Avg: {sum(exit_strengths)/len(exit_strengths):.4f}")
            
            # Count how many times signals were close to thresholds
            entry_threshold = 0.5
            exit_threshold = 0.4
            close_entries = sum(1 for s in entry_strengths if s > entry_threshold * 0.8 and s < entry_threshold)
            close_exits = sum(1 for s in exit_strengths if s > exit_threshold * 0.8 and s < exit_threshold)
            
            print(f"Signals close to but below entry threshold: {close_entries}")
            print(f"Signals close to but below exit threshold: {close_exits}")
        
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
        print("\n===== BACKTEST RESULTS =====")
        print(f"Testing Period: {hourly_data.index[0]} to {hourly_data.index[-1]}")
        print(f"Initial Capital: ${self.initial_capital:.2f}")
        print(f"Final Capital: ${final_value:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate * 100:.2f}%")
        print(f"Average Profit: ${avg_profit:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        if self.in_position:
            print(f"Currently In Position: {self.position_size:.6f} ETH @ ${self.entry_price:.2f}")
            print(f"Unrealized P&L: ${unrealized_pl:.2f}")
        
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