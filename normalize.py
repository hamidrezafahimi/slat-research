import numpy as np

def normalize(numbers):
    # Convert the input list to a NumPy array
    arr = np.array(numbers)
    
    # Calculate the minimum and maximum values of the array
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    # Apply min-max normalization
    normalized_arr = (arr - min_val) / (max_val - min_val)
    
    return normalized_arr

# Example usage
input_numbers = [10, 20, 30, 40, 50]
normalized_numbers = normalize(input_numbers)

print("Original numbers:", input_numbers)
print("Normalized numbers:", normalized_numbers)
