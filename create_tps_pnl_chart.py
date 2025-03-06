import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# Function to extract data from performance_results.txt
def extract_data(file_path):
    tps_values = []
    pnl_values = []
    uncapped_pnl_values = []  # For uncapped P&L analysis
    volume_values = []  # For trade count tracking
    usd_volume_values = []  # For USD volume tracking
    
    with open(file_path, 'r') as file:
        content = file.read()
        
        # Find all performance metrics sections
        performance_sections = re.findall(r'Performance metrics:.*?P&L metrics:.*?Total P&L: ([-\d.]+)', content, re.DOTALL)
        
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

# Function to create the TPS vs P&L chart
def create_chart(tps_values, pnl_values, output_path):
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
    
    # Save the chart
    plt.tight_layout()
    plt.savefig(output_path)
    
    # Display additional information
    print(f"Chart saved to {output_path}")
    print(f"Number of data points: {len(tps_values)}")
    print(f"TPS range: {min(tps_values)} to {max(tps_values)}")
    print(f"P&L range: ${min(pnl_values):.2f} to ${max(pnl_values):.2f}")
    print(f"Correlation coefficient: {correlation:.4f}")
    
    # Calculate linear regression equation
    slope = z[0]
    intercept = z[1]
    print(f"Linear regression equation: P&L = {slope:.4f} × TPS + {intercept:.4f}")

# Function to create the TPS vs Uncapped P&L chart
def create_uncapped_chart(tps_values, uncapped_pnl_values, output_path):
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    plt.scatter(tps_values, uncapped_pnl_values, alpha=0.7, s=50, c=tps_values, cmap='viridis')
    
    # Add trend line
    z = np.polyfit(tps_values, uncapped_pnl_values, 1)
    p = np.poly1d(z)
    plt.plot(tps_values, p(tps_values), "r--", alpha=0.7, linewidth=2)
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(tps_values, uncapped_pnl_values)[0, 1]
    
    # Add labels and title
    plt.xlabel('Transactions Per Second (TPS)', fontsize=14)
    plt.ylabel('Uncapped Total P&L', fontsize=14)
    plt.title(f'Correlation Between TPS and Uncapped Total P&L\nCorrelation Coefficient: {correlation:.4f}', fontsize=16)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Format y-axis to show dollar signs
    def currency_formatter(x, pos):
        return f'${x:.2f}'
    
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('TPS Value', fontsize=12)
    
    # Save the chart
    plt.tight_layout()
    plt.savefig(output_path)
    
    # Display additional information
    print(f"Uncapped chart saved to {output_path}")
    print(f"Number of data points: {len(tps_values)}")
    print(f"TPS range: {min(tps_values)} to {max(tps_values)}")
    print(f"Uncapped P&L range: ${min(uncapped_pnl_values):.2f} to ${max(uncapped_pnl_values):.2f}")
    print(f"Correlation coefficient: {correlation:.4f}")
    
    # Calculate linear regression equation
    slope = z[0]
    intercept = z[1]
    print(f"Linear regression equation: Uncapped P&L = {slope:.4f} × TPS + {intercept:.4f}")

# Function to create the USD Volume vs P&L chart
def create_volume_chart(usd_volume_values, pnl_values, output_path):
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
    
    # Save the chart
    plt.tight_layout()
    plt.savefig(output_path)
    
    # Display additional information
    print(f"Chart saved to {output_path}")
    print(f"Number of data points: {len(valid_volumes)}")
    print(f"USD Volume range: ${min(valid_volumes) if valid_volumes else 0:.2f} to ${max(valid_volumes) if valid_volumes else 0:.2f}")
    print(f"P&L range: ${min(valid_pnl) if valid_pnl else 0:.2f} to ${max(valid_pnl) if valid_pnl else 0:.2f}")
    print(f"Correlation coefficient: {correlation:.4f}")
    
    # Calculate linear regression equation
    slope = z[0]
    intercept = z[1]
    print(f"Linear regression equation: P&L = {slope:.4f} × USD Volume + {intercept:.4f}")

# Function to create a time series chart showing all metrics including uncapped P&L
def create_combined_time_series(tps_values, pnl_values, uncapped_pnl_values, usd_volume_values, output_path):
    plt.figure(figsize=(14, 12))
    
    # Create four subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # Plot TPS over time
    ax1.plot(range(len(tps_values)), tps_values, 'g-', linewidth=2)
    ax1.set_ylabel('TPS', fontsize=12)
    ax1.set_title('TPS Progression Over Time', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot USD Volume over time
    valid_volume_length = min(len(usd_volume_values), len(tps_values))
    ax2.plot(range(valid_volume_length), usd_volume_values[:valid_volume_length], 'm-', linewidth=2)
    ax2.set_ylabel('USD Volume', fontsize=12)
    ax2.set_title('USD Trading Volume Progression Over Time', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Format y-axis to show dollar signs for USD Volume
    def currency_formatter(x, pos):
        return f'${x:.2f}'
    
    ax2.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    
    # Plot capped P&L over time
    ax3.plot(range(len(pnl_values)), pnl_values, 'b-', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)  # Zero line
    ax3.set_ylabel('Capped P&L', fontsize=12)
    ax3.set_title('Capped P&L Progression Over Time', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Format y-axis to show dollar signs for P&L
    ax3.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    
    # Plot uncapped P&L over time
    ax4.plot(range(len(uncapped_pnl_values)), uncapped_pnl_values, 'r-', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)  # Zero line
    ax4.set_xlabel('Time (Measurement Intervals)', fontsize=12)
    ax4.set_ylabel('Uncapped P&L', fontsize=12)
    ax4.set_title('Uncapped P&L Progression Over Time', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    # Format y-axis to show dollar signs for uncapped P&L
    ax4.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    
    print(f"Combined time series chart saved to {output_path}")

if __name__ == "__main__":
    # Extract data
    tps_values, pnl_values, uncapped_pnl_values, volume_values, usd_volume_values = extract_data("performance_results.txt")
    
    # Create charts
    create_chart(tps_values, pnl_values, "charts/tps_pnl_correlation.png")
    create_uncapped_chart(tps_values, uncapped_pnl_values, "charts/tps_uncapped_pnl_correlation.png")
    create_volume_chart(usd_volume_values, pnl_values, "charts/usd_volume_pnl_correlation.png")
    create_volume_chart(usd_volume_values, uncapped_pnl_values, "charts/usd_volume_uncapped_pnl_correlation.png")
    create_combined_time_series(tps_values, pnl_values, uncapped_pnl_values, usd_volume_values, "charts/combined_time_series.png") 