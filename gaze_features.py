# gaze_features.py

"""
This script contains simple implementations of three eye-tracking features:
1. Fixation Detection
2. Gaze Dispersion Calculation
3. Saccade Detection
"""

import numpy as np

# Function to detect fixation points based on gaze data
# Parameters:
#   gaze_data : List of tuples containing (timestamp, x, y)
#   threshold : Maximum distance (in pixels) to determine fixation
# Returns:
#   List of fixations identified in the gaze data.

def detect_fixations(gaze_data, threshold=50):
    fixations = []  # Initialize an empty list to hold fixations
    current_fixation = []  # Holds current fixation points

    for i in range(len(gaze_data)):
        if i == 0:
            current_fixation.append(gaze_data[i])  # Start the first fixation
        else:
            prev_x, prev_y = current_fixation[-1][1], current_fixation[-1][2]
            current_x, current_y = gaze_data[i][1], gaze_data[i][2]
            distance = np.sqrt((current_x - prev_x) ** 2 + (current_y - prev_y) ** 2)  # Calculate distance
            
            # Check if the distance is less than the threshold
            if distance < threshold:
                current_fixation.append(gaze_data[i])  # Add point to current fixation
            else:
                if len(current_fixation) > 0:
                    fixations.append(current_fixation)  # Save current fixation if it has points
                current_fixation = [gaze_data[i]]  # Start new fixation
    if len(current_fixation) > 0:
        fixations.append(current_fixation)  # Append any remaining fixations
    return fixations


# Function to calculate gaze dispersion
# Parameters:
#   gaze_data : List of tuples containing (timestamp, x, y)
# Returns:
#   The gaze dispersion calculated as the geometric center of gaze points.

def calculate_gaze_dispersion(gaze_data):
    x_coords = [point[1] for point in gaze_data]  # Extract all x coordinates
    y_coords = [point[2] for point in gaze_data]  # Extract all y coordinates
    mean_x = np.mean(x_coords)  # Calculate average x
    mean_y = np.mean(y_coords)  # Calculate average y
    return mean_x, mean_y  # Return the center of gaze


# Function to detect saccades in gaze data
# Parameters:
#   gaze_data : List of tuples containing (timestamp, x, y)
#   threshold : Minimum distance (in pixels) for a point to be considered a saccade
# Returns:
#   List of saccades detected within the gaze data.

def detect_saccades(gaze_data, threshold=100):
    saccades = []  # Initialize an empty list to hold saccades

    for i in range(1, len(gaze_data)):
        prev_x, prev_y = gaze_data[i-1][1], gaze_data[i-1][2]
        current_x, current_y = gaze_data[i][1], gaze_data[i][2]
        distance = np.sqrt((current_x - prev_x) ** 2 + (current_y - prev_y) ** 2)  # Calculate distance
        
        if distance > threshold:
            saccades.append(gaze_data[i])  # Register a saccade if distance exceeds threshold

    return saccades  # Return list of detected saccades