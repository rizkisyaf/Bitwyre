def extract_features(row):
    """
    Extract simplified features from a single orderbook snapshot
    
    Args:
        row: DataFrame row containing orderbook data
        
    Returns:
        List of features
    """
    # Extract bid and ask data
    bid_prices = []
    bid_quantities = []
    ask_prices = []
    ask_quantities = []
    
    for j in range(1, 11):  # 10 levels
        bid_price_col = f'bid_price{j}'
        bid_quantity_col = f'bid_qty{j}'
        ask_price_col = f'ask_price{j}'
        ask_quantity_col = f'ask_qty{j}'
        
        if bid_price_col in row and not pd.isna(row[bid_price_col]):
            bid_prices.append(float(row[bid_price_col]))
            bid_quantities.append(float(row[bid_quantity_col]))
        
        if ask_price_col in row and not pd.isna(row[ask_price_col]):
            ask_prices.append(float(row[ask_price_col]))
            ask_quantities.append(float(row[ask_quantity_col]))
    
    # Skip if no bids or asks
    if len(bid_prices) == 0 or len(ask_prices) == 0:
        return None
    
    # Calculate mid price and spread
    mid_price = (bid_prices[0] + ask_prices[0]) / 2
    spread = ask_prices[0] - bid_prices[0]
    spread_pct = spread / mid_price  # Normalized spread
    
    # Calculate total volumes
    bid_volume = sum(bid_quantities)
    ask_volume = sum(ask_quantities)
    
    # Calculate imbalance
    imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-10)
    
    # Extract trade data if available
    taker_buy_volume = 0.0
    taker_sell_volume = 0.0
    trade_count = 0
    avg_trade_price = 0.0
    
    if 'taker_buy_base_volume' in row and not pd.isna(row['taker_buy_base_volume']):
        taker_buy_volume = float(row['taker_buy_base_volume'])
    
    if 'taker_sell_base_volume' in row and not pd.isna(row['taker_sell_base_volume']):
        taker_sell_volume = float(row['taker_sell_base_volume'])
    
    if 'trade_count' in row and not pd.isna(row['trade_count']):
        trade_count = float(row['trade_count'])
    
    if 'avg_trade_price' in row and not pd.isna(row['avg_trade_price']):
        avg_trade_price = float(row['avg_trade_price'])
    
    # Calculate trade-based features
    trade_imbalance = 0.0
    if taker_buy_volume + taker_sell_volume > 0:
        trade_imbalance = (taker_buy_volume - taker_sell_volume) / (taker_buy_volume + taker_sell_volume)
    
    # Calculate relative trade volume compared to orderbook depth
    relative_buy_volume = taker_buy_volume / (ask_volume + 1e-10)  # Buy orders consume ask side
    relative_sell_volume = taker_sell_volume / (bid_volume + 1e-10)  # Sell orders consume bid side
    
    # Price deviation between trades and orderbook
    price_deviation = 0.0
    if avg_trade_price > 0:
        price_deviation = (avg_trade_price - mid_price) / mid_price
    
    # Combine features (including trade data)
    features = [
        mid_price,
        spread_pct,
        imbalance,
        bid_volume,
        ask_volume,
        trade_imbalance,
        relative_buy_volume,
        relative_sell_volume,
        trade_count,
        price_deviation
    ]
    
    return features 