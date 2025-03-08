#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Integration Update Script
-------------------------------
This script updates the C++ code to work with the improved model.
It modifies the TradingModel.cpp and TradingModel.hpp files to:
1. Load normalization parameters from files
2. Apply proper normalization to features
3. Handle the model output correctly
"""

import os
import json
import argparse
import numpy as np
import re

def update_trading_model_hpp(hpp_path, input_size):
    """
    Update the TradingModel.hpp file to include normalization parameters.
    
    Parameters:
    - hpp_path: Path to TradingModel.hpp
    - input_size: Number of input features
    """
    with open(hpp_path, 'r') as f:
        content = f.read()
    
    # Add normalization parameters
    normalization_params = """
    // Normalization parameters
    std::vector<float> feature_mean_;
    std::vector<float> feature_std_;
    
    // Load normalization parameters
    bool loadNormalizationParams(const std::string& mean_path, const std::string& std_path);
    
    // Apply normalization to features
    std::vector<float> normalizeFeatures(const std::vector<float>& features);
"""
    
    # Find the private section
    private_pattern = r'private:\s*'
    if re.search(private_pattern, content):
        content = re.sub(private_pattern, 'private:\n' + normalization_params, content, count=1)
    
    # Update the constructor to accept normalization parameters
    constructor_pattern = r'TradingModel\(const std::string& model_path\);'
    new_constructor = 'TradingModel(const std::string& model_path, const std::string& mean_path = "", const std::string& std_path = "");'
    content = content.replace(constructor_pattern, new_constructor)
    
    # Update FEATURE_COUNT if needed
    feature_count_pattern = r'static constexpr int FEATURE_COUNT = \d+;'
    if re.search(feature_count_pattern, content):
        content = re.sub(feature_count_pattern, f'static constexpr int FEATURE_COUNT = {input_size};', content)
    
    with open(hpp_path, 'w') as f:
        f.write(content)
    
    print(f"Updated {hpp_path}")

def update_trading_model_cpp(cpp_path):
    """
    Update the TradingModel.cpp file to work with the improved model.
    
    Parameters:
    - cpp_path: Path to TradingModel.cpp
    """
    with open(cpp_path, 'r') as f:
        content = f.read()
    
    # Update constructor to load normalization parameters
    constructor_pattern = r'TradingModel::TradingModel\(const std::string& model_path\) \{.*?}'
    new_constructor = """TradingModel::TradingModel(const std::string& model_path, const std::string& mean_path, const std::string& std_path) {
    std::cout << "Initializing TradingModel with model path: " << model_path << std::endl;
    
    // Pre-allocate price history to avoid reallocations
    price_history_.reserve(price_history_length_);
    volatility_history_.reserve(price_history_length_);
    trade_intervals_.reserve(price_history_length_);
    
    // Initialize last trade time
    last_trade_time_ = std::chrono::high_resolution_clock::now();
    
    // Initialize normalization parameters
    price_mean_ = 0.0;
    price_std_ = 1.0;
    quantity_mean_ = 0.0;
    quantity_std_ = 1.0;
    
    // Load the model
    if (!loadModel(model_path)) {
        std::cerr << "Failed to initialize model" << std::endl;
        throw std::runtime_error("Model initialization failed");
    }
    
    // Load normalization parameters if provided
    if (!mean_path.empty() && !std_path.empty()) {
        if (!loadNormalizationParams(mean_path, std_path)) {
            std::cerr << "Failed to load normalization parameters" << std::endl;
            std::cout << "Using default normalization" << std::endl;
        }
    }
    
    std::cout << "TradingModel initialization complete" << std::endl;
}"""
    
    content = re.sub(constructor_pattern, new_constructor, content, flags=re.DOTALL)
    
    # Add method to load normalization parameters
    load_norm_params = """
bool TradingModel::loadNormalizationParams(const std::string& mean_path, const std::string& std_path) {
    try {
        std::cout << "Loading normalization parameters from:" << std::endl;
        std::cout << "  Mean: " << mean_path << std::endl;
        std::cout << "  Std: " << std_path << std::endl;
        
        // Check if files exist
        std::ifstream mean_file(mean_path.c_str(), std::ios::binary);
        std::ifstream std_file(std_path.c_str(), std::ios::binary);
        
        if (!mean_file.good() || !std_file.good()) {
            std::cerr << "Error: Normalization parameter files do not exist" << std::endl;
            return false;
        }
        
        // Read the .npy files (simplified approach)
        // In a real implementation, you would use a proper .npy file reader
        // Here we'll assume the files are in a simple binary format
        
        // Skip the .npy header (first 128 bytes should be enough)
        mean_file.seekg(128);
        std_file.seekg(128);
        
        // Allocate vectors
        feature_mean_.resize(FEATURE_COUNT);
        feature_std_.resize(FEATURE_COUNT);
        
        // Read the data
        mean_file.read(reinterpret_cast<char*>(feature_mean_.data()), FEATURE_COUNT * sizeof(float));
        std_file.read(reinterpret_cast<char*>(feature_std_.data()), FEATURE_COUNT * sizeof(float));
        
        // Check if we read the correct amount of data
        if (mean_file.fail() || std_file.fail()) {
            std::cerr << "Error reading normalization parameters" << std::endl;
            return false;
        }
        
        std::cout << "Normalization parameters loaded successfully" << std::endl;
        std::cout << "First few mean values: ";
        for (int i = 0; i < std::min(5, FEATURE_COUNT); ++i) {
            std::cout << feature_mean_[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "First few std values: ";
        for (int i = 0; i < std::min(5, FEATURE_COUNT); ++i) {
            std::cout << feature_std_[i] << " ";
        }
        std::cout << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading normalization parameters: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> TradingModel::normalizeFeatures(const std::vector<float>& features) {
    std::vector<float> normalized_features(features.size());
    
    // If normalization parameters are not loaded, return the original features
    if (feature_mean_.empty() || feature_std_.empty()) {
        return features;
    }
    
    // Apply normalization
    for (size_t i = 0; i < features.size(); ++i) {
        if (i < feature_mean_.size()) {
            // Handle NaN and Inf values
            if (std::isnan(features[i]) || std::isinf(features[i])) {
                normalized_features[i] = 0.0f;
            } else {
                // Z-score normalization
                normalized_features[i] = (features[i] - feature_mean_[i]) / feature_std_[i];
            }
        } else {
            normalized_features[i] = features[i];
        }
    }
    
    return normalized_features;
}"""
    
    # Add the new methods before the last closing brace
    content = content.rstrip()
    if content.endswith('}'):
        content = content[:-1] + load_norm_params + "\n} // namespace trading \n"
    
    # Update the predict method to use normalization
    predict_pattern = r'double TradingModel::predict\(const std::vector<float>& features\) \{.*?try \{.*?std::vector<float> cleaned_features = features;'
    new_predict = """double TradingModel::predict(const std::vector<float>& features) {
    if (!model_) {
        std::cerr << "❌ Error: Model not loaded" << std::endl;
        throw std::runtime_error("Model not loaded");
    }
    
    try {
        std::cout << "\\n📈 Feature Analysis:" << std::endl;
        std::cout << "Vector size: " << features.size() << std::endl;
        std::cout << "First 10 features: ";
        for (size_t i = 0; i < std::min(features.size(), (size_t)10); ++i) {
            std::cout << features[i] << " ";
        }
        std::cout << std::endl;
        
        // Check for invalid values in features and replace them
        bool has_invalid = false;
        std::vector<float> cleaned_features = features;"""
    
    content = re.sub(predict_pattern, new_predict, content, flags=re.DOTALL)
    
    # Update the tensor creation to use normalization
    tensor_pattern = r'auto input = torch::tensor\(cleaned_features\).reshape\(\{1, -1\}\);'
    new_tensor = """// Apply normalization
        std::vector<float> normalized_features = normalizeFeatures(cleaned_features);
        
        std::cout << "\\n🔄 Normalized features:" << std::endl;
        std::cout << "First 10 normalized features: ";
        for (size_t i = 0; i < std::min(normalized_features.size(), (size_t)10); ++i) {
            std::cout << normalized_features[i] << " ";
        }
        std::cout << std::endl;
        
        auto input = torch::tensor(normalized_features).reshape({1, -1});"""
    
    content = content.replace(tensor_pattern, new_tensor)
    
    with open(cpp_path, 'w') as f:
        f.write(content)
    
    print(f"Updated {cpp_path}")

def create_model_loader_script(output_path, model_dir):
    """
    Create a script to load the latest model and its metadata.
    
    Parameters:
    - output_path: Path to save the script
    - model_dir: Directory containing the models
    """
    script_content = f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-

\"\"\"
Model Loader Script
------------------
This script loads the latest model and its metadata, and copies it to the project directory.
\"\"\"

import os
import json
import shutil
import argparse
from datetime import datetime

def find_latest_model(model_dir):
    \"\"\"
    Find the latest model in the model directory.
    
    Parameters:
    - model_dir: Directory containing the models
    
    Returns:
    - metadata_path: Path to the latest model metadata
    \"\"\"
    metadata_files = []
    for file in os.listdir(model_dir):
        if file.startswith('model_metadata_') and file.endswith('.json'):
            metadata_files.append(os.path.join(model_dir, file))
    
    if not metadata_files:
        return None
    
    # Sort by modification time (newest first)
    metadata_files.sort(key=os.path.getmtime, reverse=True)
    return metadata_files[0]

def load_model(model_dir, target_dir):
    \"\"\"
    Load the latest model and copy it to the target directory.
    
    Parameters:
    - model_dir: Directory containing the models
    - target_dir: Directory to copy the model to
    
    Returns:
    - model_path: Path to the copied model
    - mean_path: Path to the copied mean file
    - std_path: Path to the copied std file
    \"\"\"
    metadata_path = find_latest_model(model_dir)
    if not metadata_path:
        print(f"No model found in {model_dir}")
        return None, None, None
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    model_path = metadata['model_path']
    mean_path = metadata['scaler_mean_path']
    std_path = metadata['scaler_std_path']
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy files to target directory
    target_model_path = os.path.join(target_dir, 'model.pt')
    target_mean_path = os.path.join(target_dir, 'mean.npy')
    target_std_path = os.path.join(target_dir, 'std.npy')
    
    shutil.copy2(model_path, target_model_path)
    shutil.copy2(mean_path, target_mean_path)
    shutil.copy2(std_path, target_std_path)
    
    print(f"Model copied to {target_model_path}")
    print(f"Mean file copied to {target_mean_path}")
    print(f"Std file copied to {target_std_path}")
    
    return target_model_path, target_mean_path, target_std_path

def main():
    parser = argparse.ArgumentParser(description='Load the latest model and copy it to the target directory')
    parser.add_argument('--model_dir', type=str, default='{model_dir}', help='Directory containing the models')
    parser.add_argument('--target_dir', type=str, default='.', help='Directory to copy the model to')
    
    args = parser.parse_args()
    
    model_path, mean_path, std_path = load_model(args.model_dir, args.target_dir)
    
    if model_path:
        print("\\nTo use this model in C++, update your code to load the normalization parameters:")
        print("```cpp")
        print(f'TradingModel model("{model_path}", "{mean_path}", "{std_path}");')
        print("```")

if __name__ == '__main__':
    main()
"""
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(output_path, 0o755)
    
    print(f"Created model loader script at {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Update C++ code to work with improved model')
    parser.add_argument('--cpp_path', type=str, default='../../src/model/TradingModel.cpp', help='Path to TradingModel.cpp')
    parser.add_argument('--hpp_path', type=str, default='../../include/model/TradingModel.hpp', help='Path to TradingModel.hpp')
    parser.add_argument('--input_size', type=int, default=50, help='Number of input features')
    parser.add_argument('--model_dir', type=str, default='../../models', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Update C++ files
    update_trading_model_hpp(args.hpp_path, args.input_size)
    update_trading_model_cpp(args.cpp_path)
    
    # Create model loader script
    loader_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'load_model.py')
    create_model_loader_script(loader_script_path, args.model_dir)
    
    print("\nUpdates complete!")
    print("Next steps:")
    print("1. Train a model using improved_model.py")
    print("2. Load the model using load_model.py")
    print("3. Rebuild the C++ project")
    print("4. Run the trading bot with the improved model")

if __name__ == '__main__':
    main() 