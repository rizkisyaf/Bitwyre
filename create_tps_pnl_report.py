import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import os
import base64
from io import BytesIO
import datetime

# Helper functions for correlation analysis
def get_correlation_class(correlation):
    """Return CSS class based on correlation strength."""
    if correlation > 0.3:
        return "positive"
    elif correlation < -0.3:
        return "negative"
    else:
        return "neutral"

def get_correlation_strength(correlation):
    """Return description of correlation strength."""
    abs_corr = abs(correlation)
    if abs_corr < 0.2:
        return "Very weak"
    elif abs_corr < 0.4:
        return "Weak"
    elif abs_corr < 0.6:
        return "Moderate"
    elif abs_corr < 0.8:
        return "Strong"
    else:
        return "Very strong"

def get_correlation_description(correlation):
    """Return full description of correlation."""
    strength = get_correlation_strength(correlation)
    direction = "positive" if correlation >= 0 else "negative"
    return f"{strength} {direction} correlation"

# Function to extract data from performance_results.txt
def extract_data(file_path):
    tps_values = []
    pnl_values = []
    uncapped_pnl_values = []  # For uncapped P&L analysis
    volume_values = []  # For trade count tracking
    usd_volume_values = []  # For USD volume tracking
    
    with open(file_path, 'r') as file:
        content = file.read()
        
        # Extract TPS, P&L, and volume values
        tps_pattern = re.compile(r'Trades per second: (\d+)')
        pnl_pattern = re.compile(r'Total P&L: ([-\d.]+)')
        uncapped_pnl_pattern = re.compile(r'Uncapped Total P&L: ([-\d.]+)')  # New pattern for uncapped P&L
        total_trades_pattern = re.compile(r'Total trades: (\d+)')  # For trade count tracking
        
        # Extract USD volume information directly from the bot's output
        usd_volume_pattern = re.compile(r'Total USD volume: \$([\d.]+)')
        usd_volume_matches = usd_volume_pattern.findall(content)
        
        tps_matches = tps_pattern.findall(content)
        pnl_matches = pnl_pattern.findall(content)
        uncapped_pnl_matches = uncapped_pnl_pattern.findall(content)  # Extract uncapped P&L values
        total_trades_matches = total_trades_pattern.findall(content)
        
        # Convert to numeric values
        tps_values = [int(tps) for tps in tps_matches]
        pnl_values = [float(pnl) for pnl in pnl_matches]
        
        # Convert uncapped P&L values if available
        if uncapped_pnl_matches:
            uncapped_pnl_values = [float(pnl) for pnl in uncapped_pnl_matches]
            
            # If we have more uncapped P&L values than TPS values, trim the excess
            if len(uncapped_pnl_values) > len(tps_values) and len(tps_values) > 0:
                uncapped_pnl_values = uncapped_pnl_values[:len(tps_values)]
            
            # If we have fewer uncapped P&L values than TPS values, pad with the last value
            elif len(uncapped_pnl_values) < len(tps_values) and len(uncapped_pnl_values) > 0:
                last_value = uncapped_pnl_values[-1]
                uncapped_pnl_values.extend([last_value] * (len(tps_values) - len(uncapped_pnl_values)))
        else:
            # If no uncapped P&L values are found, use regular P&L values
            uncapped_pnl_values = pnl_values.copy()
        
        # Calculate trade count volume as the difference between consecutive total trades values
        if total_trades_matches:
            total_trades = [int(trades) for trades in total_trades_matches]
            volume_values = [0]  # Start with 0 for the first measurement
            for i in range(1, len(total_trades)):
                # Calculate the difference (new trades since last measurement)
                new_volume = total_trades[i] - total_trades[i-1]
                # Ensure we don't have negative values due to test restarts
                volume_values.append(max(0, new_volume))
        
        # Use actual USD volume data from the bot
        if usd_volume_matches:
            # Convert to numeric values
            usd_volume_values = [float(vol) for vol in usd_volume_matches]
            
            # If we have more USD volume values than TPS values, trim the excess
            if len(usd_volume_values) > len(tps_values) and len(tps_values) > 0:
                usd_volume_values = usd_volume_values[:len(tps_values)]
            
            # If we have fewer USD volume values than TPS values, pad with the last value
            elif len(usd_volume_values) < len(tps_values) and len(usd_volume_values) > 0:
                last_value = usd_volume_values[-1]
                usd_volume_values.extend([last_value] * (len(tps_values) - len(usd_volume_values)))
        
        # If we still don't have USD volume data, use a placeholder
        if not usd_volume_values and tps_values:
            usd_volume_values = [0.0] * len(tps_values)
        
    # Ensure all arrays have the same length
    min_length = min(len(tps_values), len(pnl_values), len(uncapped_pnl_values), len(usd_volume_values))
    if min_length > 0:
        tps_values = tps_values[:min_length]
        pnl_values = pnl_values[:min_length]
        uncapped_pnl_values = uncapped_pnl_values[:min_length]
        usd_volume_values = usd_volume_values[:min_length]
        if len(volume_values) > min_length:
            volume_values = volume_values[:min_length]
        elif len(volume_values) < min_length:
            volume_values.extend([0] * (min_length - len(volume_values)))
    
    return tps_values, pnl_values, uncapped_pnl_values, volume_values, usd_volume_values

# Function to create the chart and return as base64 encoded string
def create_chart_base64(tps_values, pnl_values):
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    plt.scatter(tps_values, pnl_values, alpha=0.7, s=50, c=tps_values, cmap='viridis')
    
    # Add trend line
    z = np.polyfit(tps_values, pnl_values, 1)
    p = np.poly1d(z)
    plt.plot(tps_values, p(tps_values), "r--", alpha=0.7, linewidth=2)
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(tps_values, pnl_values)[0, 1]
    
    # Add labels and title
    plt.xlabel('Transactions Per Second (TPS)', fontsize=14)
    plt.ylabel('Total P&L', fontsize=14)
    plt.title(f'Correlation Between TPS and Total P&L\nCorrelation Coefficient: {correlation:.4f}', fontsize=16)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Format y-axis to show dollar signs
    def currency_formatter(x, pos):
        return f'${x:.2f}'
    
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('TPS Value', fontsize=12)
    
    # Save to BytesIO object
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode as base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_str, correlation, z[0], z[1]

# Function to create a chart showing USD volume vs P&L correlation
def create_volume_pnl_chart_base64(usd_volume_values, pnl_values):
    # Filter out entries where we don't have both volume and P&L data
    valid_indices = [i for i in range(min(len(usd_volume_values), len(pnl_values))) 
                     if i < len(usd_volume_values) and i < len(pnl_values)]
    
    valid_volumes = [usd_volume_values[i] for i in valid_indices]
    valid_pnl = [pnl_values[i] for i in valid_indices]
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    plt.scatter(valid_volumes, valid_pnl, alpha=0.7, s=50, c=valid_volumes, cmap='plasma')
    
    # Add trend line if we have enough data points
    if len(valid_volumes) > 1:
        z = np.polyfit(valid_volumes, valid_pnl, 1)
        p = np.poly1d(z)
        plt.plot(valid_volumes, p(valid_volumes), "r--", alpha=0.7, linewidth=2)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(valid_volumes, valid_pnl)[0, 1]
    else:
        correlation = 0
        z = [0, 0]
    
    # Add labels and title
    plt.xlabel('Trading Volume (USD)', fontsize=14)
    plt.ylabel('Total P&L', fontsize=14)
    plt.title(f'Correlation Between USD Trading Volume and Total P&L\nCorrelation Coefficient: {correlation:.4f}', fontsize=16)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Format y-axis to show dollar signs
    def currency_formatter(x, pos):
        return f'${x:.2f}'
    
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    
    # Format x-axis to show dollar signs
    plt.gca().xaxis.set_major_formatter(FuncFormatter(currency_formatter))
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('USD Volume', fontsize=12)
    
    # Save to BytesIO object
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode as base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_str, correlation, z[0], z[1]

# Function to create a second chart showing P&L over time
def create_time_series_chart_base64(tps_values, pnl_values):
    plt.figure(figsize=(12, 8))
    
    # Create line chart
    plt.plot(range(len(pnl_values)), pnl_values, 'b-', linewidth=2)
    plt.plot(range(len(tps_values)), [0] * len(tps_values), 'k--', alpha=0.3)  # Zero line
    
    # Add labels and title
    plt.xlabel('Time (Measurement Intervals)', fontsize=14)
    plt.ylabel('Total P&L', fontsize=14)
    plt.title('P&L Progression Over Time', fontsize=16)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Format y-axis to show dollar signs
    def currency_formatter(x, pos):
        return f'${x:.2f}'
    
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    
    # Save to BytesIO object
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode as base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_str

# Function to create a chart showing USD volume over time
def create_volume_time_series_chart_base64(usd_volume_values):
    plt.figure(figsize=(12, 8))
    
    # Create line chart
    plt.plot(range(len(usd_volume_values)), usd_volume_values, 'm-', linewidth=2)
    
    # Add labels and title
    plt.xlabel('Time (Measurement Intervals)', fontsize=14)
    plt.ylabel('Trading Volume (USD)', fontsize=14)
    plt.title('USD Trading Volume Progression Over Time', fontsize=16)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Format y-axis to show dollar signs
    def currency_formatter(x, pos):
        return f'${x:.2f}'
    
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    
    # Save to BytesIO object
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode as base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_str

# Function to create a third chart showing TPS over time
def create_tps_time_series_chart_base64(tps_values):
    plt.figure(figsize=(12, 8))
    
    # Create line chart
    plt.plot(range(len(tps_values)), tps_values, 'g-', linewidth=2)
    
    # Add labels and title
    plt.xlabel('Time (Measurement Intervals)', fontsize=14)
    plt.ylabel('Transactions Per Second (TPS)', fontsize=14)
    plt.title('TPS Progression Over Time', fontsize=16)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save to BytesIO object
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode as base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_str

# Function to create the TPS vs Uncapped P&L chart as base64
def create_uncapped_chart_base64(tps_values, uncapped_pnl_values):
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot
    plt.scatter(tps_values, uncapped_pnl_values, alpha=0.7, s=50, c=tps_values, cmap='viridis')
    
    # Add trend line
    z = np.polyfit(tps_values, uncapped_pnl_values, 1)
    p = np.poly1d(z)
    plt.plot(tps_values, p(tps_values), "r--", alpha=0.7, linewidth=2)
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(tps_values, uncapped_pnl_values)[0, 1]
    
    # Add labels and title
    plt.xlabel('Transactions Per Second (TPS)', fontsize=12)
    plt.ylabel('Uncapped Total P&L', fontsize=12)
    plt.title(f'Correlation Between TPS and Uncapped Total P&L\nCorrelation Coefficient: {correlation:.4f}', fontsize=14)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Format y-axis to show dollar signs
    def currency_formatter(x, pos):
        return f'${x:.2f}'
    
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('TPS Value', fontsize=10)
    
    # Convert to base64
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_str, correlation, z[0], z[1]

# Function to create the USD Volume vs Uncapped P&L chart as base64
def create_volume_uncapped_pnl_chart_base64(usd_volume_values, uncapped_pnl_values):
    # Filter out entries where we don't have both volume and P&L data
    valid_indices = [i for i in range(min(len(usd_volume_values), len(uncapped_pnl_values))) 
                     if i < len(usd_volume_values) and i < len(uncapped_pnl_values)]
    
    valid_volumes = [usd_volume_values[i] for i in valid_indices]
    valid_pnl = [uncapped_pnl_values[i] for i in valid_indices]
    
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot
    plt.scatter(valid_volumes, valid_pnl, alpha=0.7, s=50, c=valid_volumes, cmap='plasma')
    
    # Add trend line if we have enough data points
    correlation = 0
    slope = 0
    intercept = 0
    
    if len(valid_volumes) > 1:
        z = np.polyfit(valid_volumes, valid_pnl, 1)
        p = np.poly1d(z)
        plt.plot(valid_volumes, p(valid_volumes), "r--", alpha=0.7, linewidth=2)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(valid_volumes, valid_pnl)[0, 1]
        slope = z[0]
        intercept = z[1]
    
    # Add labels and title
    plt.xlabel('Trading Volume (USD)', fontsize=12)
    plt.ylabel('Uncapped Total P&L', fontsize=12)
    plt.title(f'Correlation Between USD Trading Volume and Uncapped Total P&L\nCorrelation Coefficient: {correlation:.4f}', fontsize=14)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Format y-axis to show dollar signs
    def currency_formatter(x, pos):
        return f'${x:.2f}'
    
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    
    # Format x-axis to show dollar signs
    plt.gca().xaxis.set_major_formatter(FuncFormatter(currency_formatter))
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('USD Volume', fontsize=10)
    
    # Convert to base64
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_str, correlation, slope, intercept

# Function to create a time series chart for uncapped P&L
def create_uncapped_pnl_time_series_chart_base64(uncapped_pnl_values):
    plt.figure(figsize=(10, 6))
    
    # Plot P&L over time
    plt.plot(range(len(uncapped_pnl_values)), uncapped_pnl_values, 'r-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)  # Zero line
    
    # Add labels and title
    plt.xlabel('Time (Measurement Intervals)', fontsize=12)
    plt.ylabel('Uncapped P&L', fontsize=12)
    plt.title('Uncapped P&L Progression Over Time', fontsize=14)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Format y-axis to show dollar signs
    def currency_formatter(x, pos):
        return f'${x:.2f}'
    
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    
    # Convert to base64
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_str

# Function to create HTML report
def create_html_report(tps_values, pnl_values, uncapped_pnl_values, volume_values, usd_volume_values, output_path):
    # Create charts
    tps_pnl_chart, tps_pnl_corr, tps_pnl_slope, tps_pnl_intercept = create_chart_base64(tps_values, pnl_values)
    
    # Create uncapped P&L charts
    tps_uncapped_pnl_chart, tps_uncapped_pnl_corr, tps_uncapped_pnl_slope, tps_uncapped_pnl_intercept = create_uncapped_chart_base64(tps_values, uncapped_pnl_values)
    
    volume_pnl_chart, volume_pnl_corr, volume_pnl_slope, volume_pnl_intercept = create_volume_pnl_chart_base64(usd_volume_values, pnl_values)
    
    # Create uncapped volume P&L chart
    volume_uncapped_pnl_chart, volume_uncapped_pnl_corr, volume_uncapped_pnl_slope, volume_uncapped_pnl_intercept = create_volume_uncapped_pnl_chart_base64(usd_volume_values, uncapped_pnl_values)
    
    tps_time_chart = create_tps_time_series_chart_base64(tps_values)
    pnl_time_chart = create_time_series_chart_base64(tps_values, pnl_values)
    
    # Create uncapped P&L time series chart
    uncapped_pnl_time_chart = create_uncapped_pnl_time_series_chart_base64(uncapped_pnl_values)
    
    volume_time_chart = create_volume_time_series_chart_base64(usd_volume_values)
    
    # Calculate statistics
    avg_tps = sum(tps_values) / len(tps_values) if tps_values else 0
    max_tps = max(tps_values) if tps_values else 0
    min_tps = min(tps_values) if tps_values else 0
    
    avg_pnl = sum(pnl_values) / len(pnl_values) if pnl_values else 0
    max_pnl = max(pnl_values) if pnl_values else 0
    min_pnl = min(pnl_values) if pnl_values else 0
    
    # Calculate uncapped P&L statistics
    avg_uncapped_pnl = sum(uncapped_pnl_values) / len(uncapped_pnl_values) if uncapped_pnl_values else 0
    max_uncapped_pnl = max(uncapped_pnl_values) if uncapped_pnl_values else 0
    min_uncapped_pnl = min(uncapped_pnl_values) if uncapped_pnl_values else 0
    
    avg_volume = sum(usd_volume_values) / len(usd_volume_values) if usd_volume_values else 0
    max_volume = max(usd_volume_values) if usd_volume_values else 0
    min_volume = min(usd_volume_values) if usd_volume_values else 0
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TPS, P&L, and USD Volume Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            h1, h2, h3 {{ color: #333; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .chart-container {{ margin-bottom: 30px; }}
            .chart {{ max-width: 100%; height: auto; }}
            .stats {{ display: flex; flex-wrap: wrap; margin-bottom: 20px; }}
            .stat-box {{ background-color: #f5f5f5; padding: 15px; margin: 10px; border-radius: 5px; flex: 1; min-width: 200px; }}
            .correlation {{ font-weight: bold; }}
            .positive {{ color: green; }}
            .negative {{ color: red; }}
            .neutral {{ color: orange; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            .timestamp {{ color: #666; font-size: 0.8em; margin-top: 50px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Trading Bot Performance Analysis</h1>
            <p>This report analyzes the relationship between Transactions Per Second (TPS), Profit and Loss (P&L), and USD Trading Volume.</p>
            
            <div class="stats">
                <div class="stat-box">
                    <h3>TPS Statistics</h3>
                    <p>Average: {avg_tps:.2f}</p>
                    <p>Maximum: {max_tps}</p>
                    <p>Minimum: {min_tps}</p>
                </div>
                <div class="stat-box">
                    <h3>P&L Statistics (Capped)</h3>
                    <p>Average: ${avg_pnl:.2f}</p>
                    <p>Maximum: ${max_pnl:.2f}</p>
                    <p>Minimum: ${min_pnl:.2f}</p>
                </div>
                <div class="stat-box">
                    <h3>P&L Statistics (Uncapped)</h3>
                    <p>Average: ${avg_uncapped_pnl:.2f}</p>
                    <p>Maximum: ${max_uncapped_pnl:.2f}</p>
                    <p>Minimum: ${min_uncapped_pnl:.2f}</p>
                </div>
                <div class="stat-box">
                    <h3>USD Volume Statistics</h3>
                    <p>Average: ${avg_volume:.2f}</p>
                    <p>Maximum: ${max_volume:.2f}</p>
                    <p>Minimum: ${min_volume:.2f}</p>
                </div>
            </div>
            
            <h2>Correlation Analysis</h2>
            
            <table>
                <tr>
                    <th>Relationship</th>
                    <th>Correlation Coefficient</th>
                    <th>Strength</th>
                    <th>Linear Equation</th>
                </tr>
                <tr>
                    <td>TPS vs P&L (Capped)</td>
                    <td class="correlation {get_correlation_class(tps_pnl_corr)}">{tps_pnl_corr:.4f}</td>
                    <td>{get_correlation_strength(tps_pnl_corr)}</td>
                    <td>P&L = {tps_pnl_slope:.4f} × TPS + {tps_pnl_intercept:.4f}</td>
                </tr>
                <tr>
                    <td>TPS vs P&L (Uncapped)</td>
                    <td class="correlation {get_correlation_class(tps_uncapped_pnl_corr)}">{tps_uncapped_pnl_corr:.4f}</td>
                    <td>{get_correlation_strength(tps_uncapped_pnl_corr)}</td>
                    <td>P&L = {tps_uncapped_pnl_slope:.4f} × TPS + {tps_uncapped_pnl_intercept:.4f}</td>
                </tr>
                <tr>
                    <td>USD Volume vs P&L (Capped)</td>
                    <td class="correlation {get_correlation_class(volume_pnl_corr)}">{volume_pnl_corr:.4f}</td>
                    <td>{get_correlation_strength(volume_pnl_corr)}</td>
                    <td>P&L = {volume_pnl_slope:.4f} × USD Volume + {volume_pnl_intercept:.4f}</td>
                </tr>
                <tr>
                    <td>USD Volume vs P&L (Uncapped)</td>
                    <td class="correlation {get_correlation_class(volume_uncapped_pnl_corr)}">{volume_uncapped_pnl_corr:.4f}</td>
                    <td>{get_correlation_strength(volume_uncapped_pnl_corr)}</td>
                    <td>P&L = {volume_uncapped_pnl_slope:.4f} × USD Volume + {volume_uncapped_pnl_intercept:.4f}</td>
                </tr>
            </table>
            
            <h2>TPS vs P&L Correlation</h2>
            
            <div class="chart-container">
                <h3>TPS vs P&L (Capped)</h3>
                <img src="data:image/png;base64,{tps_pnl_chart}" alt="TPS vs P&L Correlation" class="chart">
                <p>This chart shows the relationship between Transactions Per Second (TPS) and Profit and Loss (P&L) with a correlation coefficient of {tps_pnl_corr:.4f}, indicating a {get_correlation_description(tps_pnl_corr)}.</p>
            </div>
            
            <div class="chart-container">
                <h3>TPS vs P&L (Uncapped)</h3>
                <img src="data:image/png;base64,{tps_uncapped_pnl_chart}" alt="TPS vs Uncapped P&L Correlation" class="chart">
                <p>This chart shows the relationship between Transactions Per Second (TPS) and Uncapped Profit and Loss (P&L) with a correlation coefficient of {tps_uncapped_pnl_corr:.4f}, indicating a {get_correlation_description(tps_uncapped_pnl_corr)}.</p>
            </div>
            
            <h2>USD Volume vs P&L Correlation</h2>
            
            <div class="chart-container">
                <h3>USD Volume vs P&L (Capped)</h3>
                <img src="data:image/png;base64,{volume_pnl_chart}" alt="USD Volume vs P&L Correlation" class="chart">
                <p>This chart shows the relationship between USD Trading Volume and Profit and Loss (P&L) with a correlation coefficient of {volume_pnl_corr:.4f}, indicating a {get_correlation_description(volume_pnl_corr)}.</p>
            </div>
            
            <div class="chart-container">
                <h3>USD Volume vs P&L (Uncapped)</h3>
                <img src="data:image/png;base64,{volume_uncapped_pnl_chart}" alt="USD Volume vs Uncapped P&L Correlation" class="chart">
                <p>This chart shows the relationship between USD Trading Volume and Uncapped Profit and Loss (P&L) with a correlation coefficient of {volume_uncapped_pnl_corr:.4f}, indicating a {get_correlation_description(volume_uncapped_pnl_corr)}.</p>
            </div>
            
            <h2>Time Series Analysis</h2>
            
            <div class="chart-container">
                <h3>TPS Over Time</h3>
                <img src="data:image/png;base64,{tps_time_chart}" alt="TPS Time Series" class="chart">
                <p>This chart shows how Transactions Per Second (TPS) changed over time during the test period.</p>
            </div>
            
            <div class="chart-container">
                <h3>P&L Over Time (Capped)</h3>
                <img src="data:image/png;base64,{pnl_time_chart}" alt="P&L Time Series" class="chart">
                <p>This chart shows how Profit and Loss (P&L) changed over time during the test period.</p>
            </div>
            
            <div class="chart-container">
                <h3>P&L Over Time (Uncapped)</h3>
                <img src="data:image/png;base64,{uncapped_pnl_time_chart}" alt="Uncapped P&L Time Series" class="chart">
                <p>This chart shows how Uncapped Profit and Loss (P&L) changed over time during the test period.</p>
            </div>
            
            <div class="chart-container">
                <h3>USD Volume Over Time</h3>
                <img src="data:image/png;base64,{volume_time_chart}" alt="USD Volume Time Series" class="chart">
                <p>This chart shows how USD Trading Volume changed over time during the test period.</p>
            </div>
            
            <h2>Conclusions</h2>
            
            <p>Based on the analysis of {len(tps_values)} data points:</p>
            
            <ol>
                <li>The correlation between TPS and capped P&L is {tps_pnl_corr:.4f}, indicating a {get_correlation_description(tps_pnl_corr)}.</li>
                <li>The correlation between TPS and uncapped P&L is {tps_uncapped_pnl_corr:.4f}, indicating a {get_correlation_description(tps_uncapped_pnl_corr)}.</li>
                <li>The correlation between USD Volume and capped P&L is {volume_pnl_corr:.4f}, indicating a {get_correlation_description(volume_pnl_corr)}.</li>
                <li>The correlation between USD Volume and uncapped P&L is {volume_uncapped_pnl_corr:.4f}, indicating a {get_correlation_description(volume_uncapped_pnl_corr)}.</li>
            </ol>
            
            <p>The uncapped P&L analysis provides a more accurate view of the true relationship between trading activity and profitability, as it's not artificially limited by the risk management cap.</p>
            
            <p class="timestamp">Report generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report saved to {output_path}")

if __name__ == "__main__":
    # Extract data
    tps_values, pnl_values, uncapped_pnl_values, volume_values, usd_volume_values = extract_data("performance_results.txt")
    
    # Create HTML report
    create_html_report(tps_values, pnl_values, uncapped_pnl_values, volume_values, usd_volume_values, "charts/tps_pnl_usd_volume_analysis.html") 