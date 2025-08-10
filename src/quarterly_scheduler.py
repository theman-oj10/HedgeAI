#!/usr/bin/env python3
"""
Quarterly Scheduler

Automates quarterly portfolio debates and provides scheduling utilities.
"""

import os
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from portfolio_config import Portfolio, SAMPLE_PORTFOLIOS
from portfolio_debate import PortfolioDebateSystem
from quarterly_data_extractor import QuarterlyDataExtractor

console = Console()


class QuarterlyScheduler:
    """Manages quarterly portfolio debate scheduling and execution"""
    
    def __init__(self, config_file: str = "quarterly_config.json"):
        self.config_file = config_file
        self.portfolio_system = PortfolioDebateSystem()
        self.quarterly_extractor = QuarterlyDataExtractor()
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load quarterly scheduler configuration"""
        default_config = {
            "portfolios": [],
            "last_run_dates": {},
            "output_directory": "quarterly_reports",
            "auto_run_enabled": False,
            "notification_email": None
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load config file: {e}[/yellow]")
                return default_config
        else:
            return default_config
    
    def _save_config(self):
        """Save quarterly scheduler configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            console.print(f"[red]Error saving config: {e}[/red]")
    
    def add_portfolio(self, portfolio_name: str):
        """Add a portfolio to quarterly scheduling"""
        if portfolio_name not in self.config["portfolios"]:
            self.config["portfolios"].append(portfolio_name)
            self.config["last_run_dates"][portfolio_name] = None
            self._save_config()
            console.print(f"[green]✅ Added {portfolio_name} to quarterly schedule[/green]")
        else:
            console.print(f"[yellow]Portfolio {portfolio_name} already scheduled[/yellow]")
    
    def remove_portfolio(self, portfolio_name: str):
        """Remove a portfolio from quarterly scheduling"""
        if portfolio_name in self.config["portfolios"]:
            self.config["portfolios"].remove(portfolio_name)
            if portfolio_name in self.config["last_run_dates"]:
                del self.config["last_run_dates"][portfolio_name]
            self._save_config()
            console.print(f"[green]✅ Removed {portfolio_name} from quarterly schedule[/green]")
        else:
            console.print(f"[yellow]Portfolio {portfolio_name} not in schedule[/yellow]")
    
    def get_next_quarter_dates(self) -> tuple:
        """Get the start and end dates of the next quarter"""
        today = datetime.now()
        current_month = today.month
        current_year = today.year
        
        # Determine next quarter
        if current_month <= 3:  # Q1
            next_quarter_start = datetime(current_year, 4, 1)  # Q2
            next_quarter_end = datetime(current_year, 6, 30)
        elif current_month <= 6:  # Q2
            next_quarter_start = datetime(current_year, 7, 1)  # Q3
            next_quarter_end = datetime(current_year, 9, 30)
        elif current_month <= 9:  # Q3
            next_quarter_start = datetime(current_year, 10, 1)  # Q4
            next_quarter_end = datetime(current_year, 12, 31)
        else:  # Q4
            next_quarter_start = datetime(current_year + 1, 1, 1)  # Next year Q1
            next_quarter_end = datetime(current_year + 1, 3, 31)
        
        return next_quarter_start.strftime("%Y-%m-%d"), next_quarter_end.strftime("%Y-%m-%d")
    
    def is_quarter_end(self, buffer_days: int = 7) -> bool:
        """Check if we're within buffer_days of quarter end"""
        today = datetime.now()
        current_month = today.month
        
        # Quarter end months
        quarter_end_months = [3, 6, 9, 12]
        
        if current_month in quarter_end_months:
            # Get last day of current month
            if current_month == 12:
                next_month = datetime(today.year + 1, 1, 1)
            else:
                next_month = datetime(today.year, current_month + 1, 1)
            
            last_day_of_month = next_month - timedelta(days=1)
            days_to_quarter_end = (last_day_of_month - today).days
            
            return days_to_quarter_end <= buffer_days
        
        return False
    
    def check_portfolios_due(self) -> List[str]:
        """Check which portfolios are due for quarterly analysis"""
        due_portfolios = []
        quarter_start, quarter_end = self.quarterly_extractor.get_quarter_dates()
        
        for portfolio_name in self.config["portfolios"]:
            last_run = self.config["last_run_dates"].get(portfolio_name)
            
            if last_run is None:
                # Never run before
                due_portfolios.append(portfolio_name)
            else:
                # Check if last run was before current quarter
                last_run_date = datetime.strptime(last_run, "%Y-%m-%d")
                quarter_start_date = datetime.strptime(quarter_start, "%Y-%m-%d")
                
                if last_run_date < quarter_start_date:
                    due_portfolios.append(portfolio_name)
        
        return due_portfolios
    
    def run_quarterly_analysis(self, portfolio_name: str, save_report: bool = True) -> bool:
        """Run quarterly analysis for a specific portfolio"""
        try:
            console.print(f"\n[bold blue]🚀 Starting quarterly analysis for {portfolio_name}[/bold blue]")
            
            # Get portfolio
            if portfolio_name in SAMPLE_PORTFOLIOS:
                portfolio = SAMPLE_PORTFOLIOS[portfolio_name]
            else:
                console.print(f"[red]Unknown portfolio: {portfolio_name}[/red]")
                return False
            
            # Run quarterly analysis
            result, json_output = self.portfolio_system.analyze_portfolio_quarterly(portfolio)
            
            # Save report if requested
            if save_report:
                self._save_quarterly_report(portfolio_name, result, json_output)
            
            # Update last run date
            self.config["last_run_dates"][portfolio_name] = datetime.now().strftime("%Y-%m-%d")
            self._save_config()
            
            console.print(f"[bold green]✅ Quarterly analysis complete for {portfolio_name}![/bold green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Error in quarterly analysis for {portfolio_name}: {e}[/red]")
            return False
    
    def _save_quarterly_report(self, portfolio_name: str, result, json_output: str):
        """Save quarterly report to file"""
        # Create output directory
        output_dir = self.config["output_directory"]
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quarter_start, quarter_end = self.quarterly_extractor.get_quarter_dates()
        
        # Save JSON output
        json_filename = f"{output_dir}/quarterly_report_{portfolio_name}_{quarter_start}_to_{quarter_end}_{timestamp}.json"
        with open(json_filename, 'w') as f:
            f.write(json_output)
        
        console.print(f"[dim]📄 Quarterly report saved to {json_filename}[/dim]")
    
    def run_all_due_portfolios(self) -> Dict[str, bool]:
        """Run quarterly analysis for all due portfolios"""
        due_portfolios = self.check_portfolios_due()
        results = {}
        
        if not due_portfolios:
            console.print("[green]✅ No portfolios due for quarterly analysis[/green]")
            return results
        
        console.print(f"[bold cyan]📊 Running quarterly analysis for {len(due_portfolios)} portfolios[/bold cyan]")
        
        for portfolio_name in due_portfolios:
            results[portfolio_name] = self.run_quarterly_analysis(portfolio_name)
        
        return results
    
    def display_schedule_status(self):
        """Display current scheduling status"""
        console.print("\n[bold blue]📅 Quarterly Schedule Status[/bold blue]")
        console.print("=" * 60)
        
        # Schedule overview
        table = Table(title="Portfolio Schedule")
        table.add_column("Portfolio", style="cyan", no_wrap=True)
        table.add_column("Last Run", style="blue")
        table.add_column("Status", style="bold")
        table.add_column("Next Due", style="green")
        
        due_portfolios = self.check_portfolios_due()
        next_quarter_start, next_quarter_end = self.get_next_quarter_dates()
        
        for portfolio_name in self.config["portfolios"]:
            last_run = self.config["last_run_dates"].get(portfolio_name, "Never")
            
            if portfolio_name in due_portfolios:
                status = "[red]DUE NOW[/red]"
                next_due = "Now"
            else:
                status = "[green]Current[/green]"
                next_due = f"End of {next_quarter_end}"
            
            table.add_row(portfolio_name, last_run, status, next_due)
        
        console.print(table)
        
        # Quarter info
        quarter_start, quarter_end = self.quarterly_extractor.get_quarter_dates()
        console.print(f"\n[bold]Current Quarter Data Period:[/bold] {quarter_start} to {quarter_end}")
        console.print(f"[bold]Next Quarter:[/bold] {next_quarter_start} to {next_quarter_end}")
        console.print(f"[bold]Quarter End Approaching:[/bold] {'Yes' if self.is_quarter_end() else 'No'}")


def main():
    """CLI interface for quarterly scheduler"""
    import sys
    
    if len(sys.argv) < 2:
        console.print("[bold red]Quarterly Scheduler Usage:[/bold red]")
        console.print("1. Add portfolio to schedule:")
        console.print("   python src/quarterly_scheduler.py add <portfolio_name>")
        console.print("2. Remove portfolio from schedule:")
        console.print("   python src/quarterly_scheduler.py remove <portfolio_name>")
        console.print("3. Run quarterly analysis for specific portfolio:")
        console.print("   python src/quarterly_scheduler.py run <portfolio_name>")
        console.print("4. Run all due portfolios:")
        console.print("   python src/quarterly_scheduler.py run_all")
        console.print("5. Check schedule status:")
        console.print("   python src/quarterly_scheduler.py status")
        console.print("\nExamples:")
        console.print("   python src/quarterly_scheduler.py add tech_growth")
        console.print("   python src/quarterly_scheduler.py run tech_growth")
        console.print("   python src/quarterly_scheduler.py run_all")
        return
    
    scheduler = QuarterlyScheduler()
    command = sys.argv[1].lower()
    
    if command == "add" and len(sys.argv) > 2:
        portfolio_name = sys.argv[2]
        scheduler.add_portfolio(portfolio_name)
    
    elif command == "remove" and len(sys.argv) > 2:
        portfolio_name = sys.argv[2]
        scheduler.remove_portfolio(portfolio_name)
    
    elif command == "run" and len(sys.argv) > 2:
        portfolio_name = sys.argv[2]
        scheduler.run_quarterly_analysis(portfolio_name)
    
    elif command == "run_all":
        scheduler.run_all_due_portfolios()
    
    elif command == "status":
        scheduler.display_schedule_status()
    
    else:
        console.print("[red]Invalid command or missing arguments[/red]")
        console.print("Use: python src/quarterly_scheduler.py --help for usage")


if __name__ == "__main__":
    main()
