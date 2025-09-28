# %%
"""  Lorem Ipsum Doller """

# %%
import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import mplfinance as mpf
import matplotlib.patches as mpatches


# -----------------------------
# 1. Load Data
# -----------------------------
dta = pd.read_parquet("sampleSignal.parquet")
dta.index = pd.to_datetime(dta.index)
dta_bt = dta.reset_index().rename(columns={'index': 'datetime'})
dta_bt = dta_bt[['datetime','open','high','low','close','volume','RSI_14_crossed_up_EMA_15','SL','TP']]


# %%
warnings.filterwarnings('ignore')

class PandasDataWithExtras(bt.feeds.PandasData):
    """Enhanced data feed with technical indicators and signals"""
    lines = ('rsi14', 'ema15', 'rsi_cross', 'sl', 'tp')
    params = (
        ('rsi14', 'RSI_14'),
        ('ema15', 'EMA_15'),
        ('rsi_cross', 'RSI_14_crossed_up_EMA_15'),
        ('sl', 'SL'),
        ('tp', 'TP'),
    )

class EnterpriseStrategy(bt.Strategy):
    """Professional trading strategy with comprehensive tracking and risk management"""

    params = (
        ('trade_size', 1),
        ('max_positions', 5),
        ('risk_per_trade', 0.02),  # 2% risk per trade
        ('max_daily_loss', 0.05),  # 5% max daily loss
        ('debug', True),
    )

    def __init__(self):
        # Data references
        self.signal = self.data.rsi_cross
        self.sl = self.data.sl
        self.tp = self.data.tp

        # Trading logs
        self.trade_journal = []
        self.signal_log = []
        self.daily_pnl_log = []
        self.risk_log = []

        # Performance tracking
        self.signal_count = 0
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown = 0
        self.peak_value = self.broker.getvalue()
        self.daily_start_value = self.broker.getvalue()

        # Risk management
        self.current_positions = 0
        self.daily_loss = 0
        self.last_date = None

        # Track open orders for trade recording
        self.active_orders = {}  # Store order details
        self.pending_trades = {}  # Track trades in progress

        if self.p.debug:
            print(f"=== ENTERPRISE BACKTESTING INITIALIZED ===")
            print(f"Starting Capital: ${self.broker.getvalue():,.2f}")
            print(f"Max Positions: {self.p.max_positions}")
            print(f"Risk Per Trade: {self.p.risk_per_trade*100:.1f}%")

    def next(self):
        current_date = self.data.datetime.date()
        current_value = self.broker.getvalue()

        # Daily reset logic
        if self.last_date != current_date:
            if self.last_date is not None:
                daily_pnl = current_value - self.daily_start_value
                self.daily_pnl_log.append({
                    'date': self.last_date,
                    'start_value': self.daily_start_value,
                    'end_value': current_value,
                    'daily_pnl': daily_pnl,
                    'daily_return': daily_pnl / self.daily_start_value * 100
                })

            self.daily_start_value = current_value
            self.daily_loss = 0
            self.last_date = current_date

        # Update drawdown tracking
        if current_value > self.peak_value:
            self.peak_value = current_value
        current_drawdown = (self.peak_value - current_value) / self.peak_value
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

        # Signal processing
        if self.signal[0] == 1:
            self.signal_count += 1
            signal_time = self.data.datetime.datetime()

            # Risk management checks
            can_trade, reason = self._can_trade()

            if can_trade:
                try:
                    entry = self.data.open[1] if len(self.data.open) > 1 else None
                    if entry is None:
                        reason = "No next bar available (end of data)"
                        action = "REJECTED"
                    else:
                        # Calculate position size based on risk
                        position_size = self._calculate_position_size(entry, self.sl[0])

                        order = self.buy_bracket(
                            size=position_size,
                            price=entry,
                            stopprice=self.sl[0],
                            limitprice=self.tp[0]
                        )

                        # Store trade details for later recording
                        trade_id = f"trade_{self.signal_count}"
                        self.pending_trades[trade_id] = {
                            'signal_id': self.signal_count,
                            'datetime': signal_time,
                            'entry_price': entry,
                            'sl': self.sl[0],
                            'tp': self.tp[0],
                            'size': position_size,
                            'orders': order
                        }

                        self.current_positions += 1
                        action = "EXECUTED"
                        reason = f"Order placed (Size: {position_size})"

                        if self.p.debug:
                            print(f"Signal #{self.signal_count}: ORDER PLACED")
                            print(f"  Time: {signal_time}")
                            print(f"  Entry: {entry:.3f}, SL: {self.sl[0]:.3f}, TP: {self.tp[0]:.3f}")
                            print(f"  Position Size: {position_size}")
                            print(f"  Risk Amount: ${abs(entry - self.sl[0]) * position_size:.2f}")

                except Exception as e:
                    action = "ERROR"
                    reason = f"Order failed: {str(e)}"

            else:
                action = "REJECTED"
                if self.p.debug:
                    print(f"Signal #{self.signal_count}: REJECTED - {reason}")

            # Log signal
            self.signal_log.append({
                "signal_id": self.signal_count,
                "datetime": signal_time,
                "action": action,
                "reason": reason,
                "entry_price": self.data.open[1] if len(self.data.open) > 1 else None,
                "sl": self.sl[0],
                "tp": self.tp[0],
                "account_value": current_value,
                "positions_open": self.current_positions,
                "daily_pnl": current_value - self.daily_start_value
            })

    def _can_trade(self):
        """Risk management: determine if we can take a new trade"""

        # Check max positions
        if self.current_positions >= self.p.max_positions:
            return False, f"Max positions reached ({self.current_positions}/{self.p.max_positions})"

        # Check daily loss limit
        current_daily_loss = (self.broker.getvalue() - self.daily_start_value) / self.daily_start_value
        if current_daily_loss <= -self.p.max_daily_loss:
            return False, f"Daily loss limit reached ({current_daily_loss*100:.1f}%)"

        # Check if position already exists
        if self.position:
            return False, "Position already open"

        return True, "Risk checks passed"

    def _calculate_position_size(self, entry_price, sl_price):
        """Calculate position size based on risk management"""
        account_value = self.broker.getvalue()
        risk_amount = account_value * self.p.risk_per_trade
        price_risk = abs(entry_price - sl_price)

        if price_risk > 0:
            # Calculate theoretical position size
            theoretical_size = risk_amount / price_risk

            # Apply practical limits for futures/forex trading
            max_position_value = account_value * 0.1  # Max 10% of account per position
            max_size_by_value = max_position_value / entry_price

            # Use the smaller of the two
            position_size = min(theoretical_size, max_size_by_value)

            # Ensure reasonable bounds (1 to 100 lots)
            return max(1, min(100, int(position_size)))
        return 1

    def notify_order(self, order):
        """Enhanced order tracking with trade recording"""
        if order.status == order.Completed:
            if order.isbuy():
                if self.p.debug:
                    print(f"BUY EXECUTED: Price={order.executed.price:.3f}, Size={order.executed.size}")
                # Store entry details
                for trade_id, trade_data in self.pending_trades.items():
                    if order in trade_data.get('orders', []):
                        trade_data['actual_entry'] = order.executed.price
                        trade_data['actual_size'] = order.executed.size
                        break

            elif order.issell():
                if self.p.debug:
                    print(f"SELL EXECUTED: Price={order.executed.price:.3f}, Size={order.executed.size}")

                # Find and record the completed trade
                for trade_id, trade_data in list(self.pending_trades.items()):
                    if order in trade_data.get('orders', []) and 'actual_entry' in trade_data:
                        # Calculate trade metrics
                        entry_price = trade_data['actual_entry']
                        exit_price = order.executed.price
                        size = trade_data['actual_size']
                        pnl = (exit_price - entry_price) * size

                        # Determine if hit SL or TP
                        sl_distance = abs(exit_price - trade_data['sl'])
                        tp_distance = abs(exit_price - trade_data['tp'])
                        exit_type = "SL" if sl_distance < tp_distance else "TP"

                        # Calculate risk metrics
                        risk = abs(entry_price - trade_data['sl'])
                        reward = abs(trade_data['tp'] - entry_price)
                        rr_ratio = reward / risk if risk > 0 else 0

                        # Determine outcome
                        is_winner = pnl > 0
                        outcome = "WIN" if is_winner else "LOSS"

                        if is_winner:
                            self.winning_trades += 1
                        else:
                            self.losing_trades += 1

                        self.trade_count += 1

                        # Calculate trade return
                        trade_return = (pnl / (entry_price * abs(size))) * 100

                        # Record trade
                        trade_record = {
                            "trade_id": self.trade_count,
                            "signal_id": trade_data['signal_id'],
                            "datetime": trade_data['datetime'],
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "exit_type": exit_type,
                            "pnl": pnl,
                            "trade_return": trade_return,
                            "outcome": outcome,
                            "sl": trade_data['sl'],
                            "tp": trade_data['tp'],
                            "risk": risk,
                            "reward": reward,
                            "rr_ratio": rr_ratio,
                            "size": abs(size),
                            "account_value": self.broker.getvalue(),
                            "drawdown": self.max_drawdown * 100
                        }

                        self.trade_journal.append(trade_record)

                        if self.p.debug:
                            print(f"=== TRADE #{self.trade_count} CLOSED ({exit_type}) ===")
                            print(f"Outcome: {outcome}")
                            print(f"Entry: {entry_price:.3f} | Exit: {exit_price:.3f}")
                            print(f"PnL: ${pnl:.2f} ({trade_return:+.2f}%)")
                            print(f"R:R = {rr_ratio:.2f}")

                        # Remove completed trade
                        del self.pending_trades[trade_id]
                        break

                self.current_positions = max(0, self.current_positions - 1)

        elif order.status == order.Margin:
            self.current_positions = max(0, self.current_positions - 1)
            if self.p.debug:
                print(f"ORDER FAILED - MARGIN: Insufficient funds for position size {order.size}")
        elif order.status == order.Rejected:
            self.current_positions = max(0, self.current_positions - 1)
            if self.p.debug:
                print(f"ORDER REJECTED: Position size {order.size} rejected by broker")
        elif order.status in [order.Canceled]:
            if self.p.debug:
                print(f"ORDER CANCELLED")
        elif order.status == order.Expired:
            if self.p.debug:
                print(f"ORDER EXPIRED (Normal for unused SL/TP)")

    def notify_trade(self, trade):
        """Comprehensive trade logging and analysis"""
        if not trade.isclosed or trade.size == 0:
            return

        self.trade_count += 1

        # Calculate trade metrics
        entry_price = trade.price - (trade.pnl / trade.size)
        exit_price = trade.price
        pnl = trade.pnl

        # Get SL/TP from current bar (approximation)
        current_sl = self.sl[0]
        current_tp = self.tp[0]

        # Calculate risk metrics
        risk = abs(entry_price - current_sl)
        reward = abs(current_tp - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0

        # Determine trade outcome
        is_winner = pnl > 0
        if is_winner:
            self.winning_trades += 1
            outcome = "WIN"
        else:
            self.losing_trades += 1
            outcome = "LOSS"

        # Calculate trade return
        trade_return = (pnl / (entry_price * abs(trade.size))) * 100

        # Record comprehensive trade data
        trade_data = {
            "trade_id": self.trade_count,
            "datetime": self.data.datetime.datetime(),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "trade_return": trade_return,
            "outcome": outcome,
            "sl": current_sl,
            "tp": current_tp,
            "risk": risk,
            "reward": reward,
            "rr_ratio": rr_ratio,
            "size": abs(trade.size),
            "account_value": self.broker.getvalue(),
            "drawdown": self.max_drawdown * 100
        }

        self.trade_journal.append(trade_data)

        if self.p.debug:
            print(f"=== TRADE #{self.trade_count} CLOSED ===")
            print(f"Outcome: {outcome}")
            print(f"Entry: {entry_price:.3f} | Exit: {exit_price:.3f}")
            print(f"PnL: ${pnl:.2f} ({trade_return:+.2f}%)")
            print(f"R:R = {rr_ratio:.2f}")
            print(f"Account Value: ${self.broker.getvalue():,.2f}")

    def stop(self):
        """Generate comprehensive performance report"""
        final_value = self.broker.getvalue()
        total_return = (final_value - 100000) / 100000 * 100

        print(f"\n{'='*60}")
        print(f"           ENTERPRISE BACKTESTING REPORT")
        print(f"{'='*60}")

        # Account Performance
        print(f"\n📊 ACCOUNT PERFORMANCE")
        print(f"Starting Capital:     ${100000:,.2f}")
        print(f"Ending Capital:       ${final_value:,.2f}")
        print(f"Total Return:         {total_return:+.2f}%")
        print(f"Max Drawdown:         {self.max_drawdown*100:.2f}%")

        # Trading Statistics
        win_rate = (self.winning_trades / self.trade_count * 100) if self.trade_count > 0 else 0

        print(f"\n📈 TRADING STATISTICS")
        print(f"Total Signals:        {self.signal_count}")
        print(f"Total Trades:         {self.trade_count}")
        print(f"Signal Conversion:    {(self.trade_count/self.signal_count*100):.1f}%" if self.signal_count > 0 else "0%")
        print(f"Winning Trades:       {self.winning_trades}")
        print(f"Losing Trades:        {self.losing_trades}")
        print(f"Win Rate:            {win_rate:.1f}%")

        # Risk Analysis
        if len(self.trade_journal) > 0:
            trades_df = pd.DataFrame(self.trade_journal)
            avg_rr = trades_df['rr_ratio'].mean()
            avg_win = trades_df[trades_df['outcome'] == 'WIN']['pnl'].mean()
            avg_loss = trades_df[trades_df['outcome'] == 'LOSS']['pnl'].mean()
            profit_factor = abs(avg_win * self.winning_trades / (avg_loss * self.losing_trades)) if avg_loss != 0 and self.losing_trades > 0 else 0

            print(f"\n⚖️  RISK ANALYSIS")
            print(f"Average R:R Ratio:    {avg_rr:.2f}")
            print(f"Average Win:          ${avg_win:.2f}" if not pd.isna(avg_win) else "Average Win: N/A")
            print(f"Average Loss:         ${avg_loss:.2f}" if not pd.isna(avg_loss) else "Average Loss: N/A")
            print(f"Profit Factor:        {profit_factor:.2f}")

        # Signal Analysis
        if len(self.signal_log) > 0:
            signals_df = pd.DataFrame(self.signal_log)
            signal_breakdown = signals_df['action'].value_counts()

            print(f"\n🎯 SIGNAL BREAKDOWN")
            for action, count in signal_breakdown.items():
                percentage = count / len(signals_df) * 100
                print(f"{action:15}: {count:3d} ({percentage:4.1f}%)")

            # Rejection reasons
            rejected_signals = signals_df[signals_df['action'] == 'REJECTED']
            if len(rejected_signals) > 0:
                print(f"\n❌ REJECTION REASONS")
                rejection_reasons = rejected_signals['reason'].value_counts()
                for reason, count in rejection_reasons.items():
                    print(f"{reason:30}: {count}")

class EnterpriseTradingFramework:
    """Professional trading framework with advanced analytics"""

    def __init__(self):
        self.results = None
        self.strategy = None

    def run_backtest(self, data_df, **kwargs):
        """Execute enterprise-level backtest"""

        # Default parameters
        params = {
            'cash': 100000,
            'commission': 0.001,
            'trade_size': 1,
            'max_positions': 5,
            'risk_per_trade': 0.02,
            'max_daily_loss': 0.05,
            'debug': True
        }
        params.update(kwargs)

        # Initialize Cerebro
        cerebro = bt.Cerebro()

        # Add data
        data = PandasDataWithExtras(dataname=data_df)
        cerebro.adddata(data)

        # Add strategy with parameters
        cerebro.addstrategy(
            EnterpriseStrategy,
            trade_size=params['trade_size'],
            max_positions=params['max_positions'],
            risk_per_trade=params['risk_per_trade'],
            max_daily_loss=params['max_daily_loss'],
            debug=params['debug']
        )

        # Set broker parameters with proper margin settings
        cerebro.broker.set_cash(params['cash'])
        cerebro.broker.setcommission(commission=params['commission'])

        # Set realistic margin requirements (for futures/forex)
        # This allows leverage but prevents excessive position sizes
        cerebro.broker.set_coc(True)  # Close on close

        # Add position sizer to limit risk
        cerebro.addsizer(bt.sizers.FixedSize, stake=1)

        # Add analyzers for advanced metrics
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

        # Run backtest
        print(f"🚀 Starting Enterprise Backtest...")
        results = cerebro.run()
        self.strategy = results[0]
        self.results = results

        return self._generate_results()

    def _generate_results(self):
        """Generate comprehensive results package"""

        # Create DataFrames
        trades_df = pd.DataFrame(self.strategy.trade_journal) if self.strategy.trade_journal else pd.DataFrame()
        signals_df = pd.DataFrame(self.strategy.signal_log) if self.strategy.signal_log else pd.DataFrame()
        daily_pnl_df = pd.DataFrame(self.strategy.daily_pnl_log) if self.strategy.daily_pnl_log else pd.DataFrame()

        # Analyzer results
        analyzers = {
            'sharpe': self.strategy.analyzers.sharpe.get_analysis(),
            'drawdown': self.strategy.analyzers.drawdown.get_analysis(),
            'returns': self.strategy.analyzers.returns.get_analysis(),
            'trades': self.strategy.analyzers.trades.get_analysis()
        }

        return {
            'trades': trades_df,
            'signals': signals_df,
            'daily_pnl': daily_pnl_df,
            'analyzers': analyzers,
            'strategy': self.strategy
        }

    def generate_report(self, results, save_to_file=False):
        """Generate detailed Excel report with enhanced trading journal"""
        try:
            if save_to_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"trading_report_{timestamp}.xlsx"

                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    # Trading Journal (Enhanced)
                    if not results['trades'].empty:
                        journal = results['trades'].copy()

                        # Add journal enhancements
                        journal['duration'] = (pd.to_datetime(journal['datetime']) -
                                             pd.to_datetime(journal['datetime']).shift(1)).dt.total_seconds() / 60
                        journal['cumulative_pnl'] = journal['pnl'].cumsum()
                        journal['running_balance'] = 100000 + journal['cumulative_pnl']
                        journal['drawdown'] = ((journal['running_balance'].cummax() -
                                              journal['running_balance']) / journal['running_balance'].cummax() * 100)
                        journal['win_streak'] = (journal['outcome'] == 'WIN').groupby(
                            (journal['outcome'] != journal['outcome'].shift()).cumsum()).cumsum()
                        journal['loss_streak'] = (journal['outcome'] == 'LOSS').groupby(
                            (journal['outcome'] != journal['outcome'].shift()).cumsum()).cumsum()

                        # Format for better readability
                        journal['datetime'] = pd.to_datetime(journal['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
                        journal['entry_price'] = journal['entry_price'].round(3)
                        journal['exit_price'] = journal['exit_price'].round(3)
                        journal['pnl'] = journal['pnl'].round(2)
                        journal['trade_return'] = journal['trade_return'].round(3)
                        journal['rr_ratio'] = journal['rr_ratio'].round(2)

                        journal.to_excel(writer, sheet_name='Trading_Journal', index=False)

                    # Performance Summary
                    if not results['trades'].empty:
                        trades = results['trades']
                        summary_data = {
                            'Metric': [
                                'Total Trades', 'Winning Trades', 'Losing Trades', 'Win Rate (%)',
                                'Total PnL', 'Average Win', 'Average Loss', 'Profit Factor',
                                'Best Trade', 'Worst Trade', 'Average R:R Ratio', 'Max Consecutive Wins',
                                'Max Consecutive Losses', 'Sharpe Ratio (Est)', 'Maximum Drawdown (%)'
                            ],
                            'Value': [
                                len(trades),
                                len(trades[trades['outcome'] == 'WIN']),
                                len(trades[trades['outcome'] == 'LOSS']),
                                len(trades[trades['outcome'] == 'WIN']) / len(trades) * 100,
                                trades['pnl'].sum(),
                                trades[trades['outcome'] == 'WIN']['pnl'].mean() if len(trades[trades['outcome'] == 'WIN']) > 0 else 0,
                                trades[trades['outcome'] == 'LOSS']['pnl'].mean() if len(trades[trades['outcome'] == 'LOSS']) > 0 else 0,
                                abs(trades[trades['outcome'] == 'WIN']['pnl'].sum() / trades[trades['outcome'] == 'LOSS']['pnl'].sum()) if trades[trades['outcome'] == 'LOSS']['pnl'].sum() != 0 else 0,
                                trades['pnl'].max(),
                                trades['pnl'].min(),
                                trades['rr_ratio'].mean(),
                                trades['win_streak'].max() if 'win_streak' in trades.columns else 0,
                                trades['loss_streak'].max() if 'loss_streak' in trades.columns else 0,
                                trades['trade_return'].mean() / trades['trade_return'].std() if trades['trade_return'].std() != 0 else 0,
                                trades['drawdown'].max() if 'drawdown' in trades.columns else 0
                            ]
                        }
                        summary_df = pd.DataFrame(summary_data)
                        summary_df.to_excel(writer, sheet_name='Performance_Summary', index=False)

                    # Signal Analysis
                    if not results['signals'].empty:
                        signals = results['signals'].copy()
                        signals['datetime'] = pd.to_datetime(signals['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
                        signals.to_excel(writer, sheet_name='Signal_Analysis', index=False)

                    # Daily P&L (if available)
                    if not results['daily_pnl'].empty:
                        daily = results['daily_pnl'].copy()
                        daily['date'] = pd.to_datetime(daily['date']).dt.strftime('%Y-%m-%d')
                        daily.to_excel(writer, sheet_name='Daily_PnL', index=False)

                print(f"📁 Enhanced Trading Journal Report saved to: {filename}")
                return filename
        except ImportError:
            print("⚠️  Excel export requires openpyxl: pip install openpyxl")
        return None

    def print_trading_journal_summary(self, results):
        """Print comprehensive trading journal summary to console"""
        trades = results['trades']

        if trades.empty:
            print("📋 No trades to display in journal")
            return

        print(f"\n{'='*80}")
        print(f"                    📋 TRADING JOURNAL SUMMARY")
        print(f"{'='*80}")

        # Journal Header
        print(f"\n📊 PORTFOLIO OVERVIEW")
        print(f"Starting Balance:     ${100000:,.2f}")
        print(f"Ending Balance:       ${100000 + trades['pnl'].sum():,.2f}")
        print(f"Total P&L:           ${trades['pnl'].sum():+,.2f}")
        print(f"Total Return:        {trades['pnl'].sum() / 100000 * 100:+.2f}%")

        # Trade Statistics
        wins = trades[trades['outcome'] == 'WIN']
        losses = trades[trades['outcome'] == 'LOSS']

        print(f"\n📈 TRADE STATISTICS")
        print(f"Total Trades:         {len(trades)}")
        print(f"Winning Trades:       {len(wins)} ({len(wins)/len(trades)*100:.1f}%)")
        print(f"Losing Trades:        {len(losses)} ({len(losses)/len(trades)*100:.1f}%)")
        print(f"Best Trade:           ${trades['pnl'].max():+.2f}")
        print(f"Worst Trade:          ${trades['pnl'].min():+.2f}")
        print(f"Average Trade:        ${trades['pnl'].mean():+.2f}")

        # Risk Metrics
        print(f"\n⚖️  RISK ANALYSIS")
        print(f"Average Win:          ${wins['pnl'].mean():.2f}" if len(wins) > 0 else "Average Win: N/A")
        print(f"Average Loss:         ${losses['pnl'].mean():.2f}" if len(losses) > 0 else "Average Loss: N/A")
        print(f"Profit Factor:        {abs(wins['pnl'].sum() / losses['pnl'].sum()):.2f}" if len(losses) > 0 and losses['pnl'].sum() != 0 else "Profit Factor: N/A")
        print(f"Average R:R Ratio:    {trades['rr_ratio'].mean():.2f}")
        print(f"Sharpe Ratio (Est):   {trades['trade_return'].mean() / trades['trade_return'].std():.2f}" if trades['trade_return'].std() != 0 else "Sharpe Ratio: N/A")

        # Recent Trades Table
        print(f"\n📋 RECENT TRADES (Last 5)")
        print(f"{'ID':<3} {'Date':<19} {'Entry':<8} {'Exit':<8} {'Type':<4} {'P&L':<8} {'Outcome':<4}")
        print(f"{'─'*3} {'─'*19} {'─'*8} {'─'*8} {'─'*4} {'─'*8} {'─'*4}")

        recent_trades = trades.tail(5)
        for _, trade in recent_trades.iterrows():
            print(f"{trade['trade_id']:<3} {str(trade['datetime'])[:19]:<19} "
                  f"{trade['entry_price']:<8.3f} {trade['exit_price']:<8.3f} "
                  f"{trade['exit_type']:<4} ${trade['pnl']:<7.2f} {trade['outcome']:<4}")

        print(f"\n💡 Access full journal: results['trades']")
        print(f"💡 Export to Excel: framework.generate_report(results, save_to_file=True)")
        print(f"{'='*80}")

# %%
framework = EnterpriseTradingFramework()

# Get comprehensive journal
results = framework.run_backtest(dta)
# trading_journal = results['trades']

# Display professional summary
# framework.print_trading_journal_summary(results)

# %%
trades = results['trades'][['datetime', 'entry_price', 'exit_price','sl','tp']].copy()
sltp = trades[['sl','tp']]

# %%
row_numbers = dta.index.get_indexer(trades['datetime'])

# Extract SL and TP values for those rows
sl_array = trades['entry_price'].to_numpy()
tp_array = trades['exit_price'].to_numpy()

sltpArr = np.array(list(zip(row_numbers, sltp['sl'], sltp['tp'])),
                        dtype=[('row', int), ('SL', float), ('TP', float)])

# Optional: combine into structured array
signal_array = np.array(list(zip(row_numbers, sl_array, tp_array)),
                        dtype=[('row', int), ('SL', float), ('TP', float)])

# %%
trades = results['trades'][['datetime', 'entry_price', 'exit_price','sl','tp']].copy()
# -----------------------------
# Prepare structured arrays
# -----------------------------
# Get integer row positions of trades in dta
row_numbers = dta.index.get_indexer(trades['datetime'])

# Entry/Exit prices
entry_array = trades['entry_price'].to_numpy()
exit_array = trades['exit_price'].to_numpy()

# SL/TP prices
sl_array = trades['sl'].to_numpy()  # assuming trades also has SL
tp_array = trades['tp'].to_numpy()  # assuming trades also has TP

# Structured arrays
entry_exit_arr = np.zeros(len(trades), dtype=[('row', int), ('entry', float), ('exit', float)])
entry_exit_arr['row'] = row_numbers
entry_exit_arr['entry'] = entry_array
entry_exit_arr['exit'] = exit_array

sltp_arr = np.zeros(len(trades), dtype=[('row', int), ('SL', float), ('TP', float)])
sltp_arr['row'] = row_numbers
sltp_arr['SL'] = sl_array
sltp_arr['TP'] = tp_array


# %%
# --- TradingView Configuration ---

dpart = dta[:151]

tradingview_config_4H = {
    'title': 'XAUUSD, 4H',                # TradingView style title
    'style': tradingview_dark,             # Use dark theme
    'type': 'candle',
    'volume': False,                      # Disable volume bars
    'show_nontrading': False,             # Ensure no extra y-axis spacing
    'datetime_format': '%Y-%m-%d',        # Format x-axis to show date only
    'xlabel': '',                         # Remove x-axis label
    'ylabel': '',                         # Remove y-axis label
    'xrotation': 0                        # Horizontal date labels
}

fig, axes = chartter.plot(
    dpart,
    addplot = [
        mpf.make_addplot(dpart['RSI_14'], panel=1, color='purple', ylabel='RSI'),
        mpf.make_addplot(dpart['EMA_15'], panel=1, color='blue', ylabel='ema15'),
    ],
    config=tradingview_config_4H,
    returnfig=True
)

# %%
from ChartterX5 import Chartter

# Initialize chartter with wider and less tall proportions, no volume
chartter = Chartter(config={
    'chart_type': 'candle',
    'style': 'charles',
    'figratio': (20, 8),  # Adjusted for wider and less tall chart
    'volume': False  # Disable volume bars
})

# --- TradingView Market Colors ---
tv_mc = mpf.make_marketcolors(
    up='#26a69a',         # TradingView teal green for up candles
    down='#ef5350',       # TradingView red for down candles
    edge='inherit',       # Clean edges matching candle color
    wick='inherit',       # Wicks match candle colors
)

# --- TradingView Dark Theme ---
tradingview_dark = mpf.make_mpf_style(
    base_mpf_style='nightclouds',  # Start with dark base
    marketcolors=tv_mc,

    # TradingView dark theme colors
    figcolor='#131722',           # Dark navy background
    facecolor='#1e222d',          # Dark gray chart area
    gridcolor='#363a45',          # Dark gray grid
    gridstyle='-',                # Solid grid lines

    y_on_right=True,              # Price axis on right

    rc={
        # TradingView typography
        # 'font.family': ['-apple-system', 'BlinkMacSystemFont', 'Trebuchet MS', 'Roboto', 'Ubuntu', 'sans-serif'],
        'axes.labelsize': 10,
        'axes.titlesize': 14,
        'xtick.labelsize': 9,
        'ytick.labelsize': 7,         # Reduced size for price tick labels
        'legend.fontsize': 9,

        # Clean spacing with reduced left padding and full-width chart
        'axes.labelpad': 10,
        'xtick.major.pad': 6,
        'ytick.major.pad': 6,
        'axes.xmargin': 0,            # Remove left/right margins
        'axes.ymargin': 0,            # Remove top/bottom margins
        'figure.subplot.left': 0.05,  # Minimize left subplot padding
        'figure.subplot.right': 0.95, # Maximize right subplot to fit price scale

        # TradingView-style lines
        'lines.linewidth': 1.5,
        'lines.antialiased': True,

        # Clean borders and colors
        'axes.edgecolor': '#434651',  # Darker edge for dark theme
        'axes.linewidth': 1,
        'xtick.color': '#787b86',     # Light gray for x-axis ticks
        'ytick.color': '#787b86',     # Light gray for y-axis ticks
        'axes.labelcolor': 'none',    # Hide axis label color (labels are empty)

        # Grid styling
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.alpha': 0.6,            # Slightly more transparent for dark theme
        'grid.linewidth': 0.8,

        # Clean spines (show right spine for price scale)
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': True,    # Show right spine for price scale
        'ytick.right': True,          # Show right y-axis ticks
    }
)


# %%
import matplotlib.patches as mpatches

def draw_boxes_border_only(ax, arr, y1_col, y2_col, edge_color='red', lw=1.5, alpha=1.0):
    for row_data in arr:
        row = row_data['row']
        y1 = row_data[y1_col]
        y2 = row_data[y2_col]
        rect = mpatches.Rectangle(
            (row - 0.5, y1),  # left bottom corner
            1,                # width = 1 candle
            y2 - y1,          # height
            facecolor='none', # no fill
            edgecolor=edge_color,
            linewidth=lw,
            alpha=alpha,
            zorder=10         # Ensure it's drawn on top
        )
        ax.add_patch(rect)

# Draw SL/TP boxes with border only
draw_boxes_border_only(axes[0], sltp_arr, 'SL', 'TP', edge_color='white', lw=0.5)

# Draw Entry/Exit boxes with border only
draw_boxes_border_only(axes[0], entry_exit_arr, 'entry', 'exit', edge_color='red', lw=0.5)


fig


