from flask import Flask, jsonify
import requests
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


app = Flask(__name__)
def data_pull(ORCHARD_NAME):
	API_BASE_URL = "https://api.aerobotics.com/farming"
	API_HEADERS = {
		'Authorization': 'Bearer 5d03db72854d43a8ce0c63e0d4fb4a261bc29b95ea46b541f537dbf0891b45d6',
		'Accept': 'application/json'
	}



	# Get surveys for orchard
	survey_response = requests.request(
		"GET", f"{API_BASE_URL}/surveys/?orchard_id={ORCHARD_NAME}", headers=API_HEADERS
	)
	surveys = survey_response.json()["results"]  # TODO: Need to handle pagination
	surveys_sorted_by_date = sorted(surveys, key=lambda survey: survey["date"])
	latest_survey_id = surveys_sorted_by_date[-1]["id"]

	# Get the trees for the latest survey
	tree_surveys_response = requests.request(
		"GET", f"{API_BASE_URL}/surveys/{latest_survey_id}/tree_surveys/", headers=API_HEADERS
	)

	tree_surveys = tree_surveys_response.json()
	results = tree_surveys.get('results', [])
	df=pd.DataFrame(results)
	return df

def compute_radius(df, target_mode=4):
    points = df[['lat', 'lng']].values
    distances = distance.cdist(points, points, metric='euclidean')
    
    radii = []
    for i, point in enumerate(points):
        dists_from_point = distances[i]
        sorted_dists = np.sort(dists_from_point)
        enclosed_points = np.sum(sorted_dists <= sorted_dists[target_mode], axis=0) - 1
        radii.append(sorted_dists[target_mode])
    
    mode_radius = np.mean(radii)
    print(f"Estimated radius: {mode_radius}")
    
    return mode_radius

def find_overlap_centers(df, radius, target_mode=4):
    points = df[['lat', 'lng']].values
    circles = []
    overlap_centers = []
    
    # Identify circles that do not enclose exactly 4 points
    for i, point in enumerate(points):
        dists_from_point = distance.cdist([point], points, metric='euclidean').flatten()
        enclosed_points_count = np.sum(dists_from_point <= radius) - 1
        if enclosed_points_count != target_mode:
            circle_points = get_circle_points(point[0], point[1], radius)
            circles.append((Polygon(circle_points), enclosed_points_count))
    
    if len(circles) < 3:
        print("Not enough circles to find overlap.")
        return overlap_centers
    
    # Find circles that have exactly 4 enclosed points
    circles_with_target_mode = [circle for circle, count in circles if count == target_mode]
    
    # Check overlaps for circles with less than 4 enclosed points
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            for k in range(j+1, len(circles)):
                combined_polygon = circles[i][0].intersection(circles[j][0]).intersection(circles[k][0])
                
                if not combined_polygon.is_empty:
                    # Discount overlaps if intersected by a circle with exactly 4 enclosed points
                    if not any(circles_with_target_mode[i].intersection(combined_polygon).is_empty for i in range(len(circles_with_target_mode))):
                        overlap_center = combined_polygon.centroid
                        overlap_centers.append(overlap_center)
    
    return overlap_centers

def evaluate_candidates(candidates, df, threshold=0.93):
    points = df[['lat', 'lng']].values
    distances = distance.cdist(points, points, metric='euclidean')
    
    # Calculate the mean nearest neighbor distance
    nearest_distances = np.min(distances + np.eye(len(distances)) * np.max(distances), axis=1)
    mean_nn_distance = np.mean(nearest_distances)
    
    # Evaluate each candidate
    valid_candidates = []
    for candidate in candidates:
        candidate_point = np.array([candidate.y, candidate.x]).reshape(1, -1)
        dists_from_candidate = distance.cdist(candidate_point, points, metric='euclidean').flatten()
        nearest_distance = np.min(dists_from_candidate)
        if nearest_distance >= threshold * mean_nn_distance:
            valid_candidates.append(candidate)
    
    return valid_candidates

def get_circle_points(lat, lng, radius, num_points=100):
    angles = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(angles) + lng
    y = radius * np.sin(angles) + lat
    return np.vstack((x, y)).T

def group_candidates_within_circles(valid_candidates, radius):
    candidate_points = np.array([[c.y, c.x] for c in valid_candidates])
    updated_candidates = []
    used = np.zeros(len(valid_candidates), dtype=bool)
    
    while not all(used):
        # Find the first unused candidate
        i = np.where(~used)[0][0]
        center_point = candidate_points[i]
        circle_points = get_circle_points(center_point[0], center_point[1], radius)
        circle_polygon = Polygon(circle_points)
        
        # Find all points within this circle
        dists = distance.cdist([center_point], candidate_points, metric='euclidean').flatten()
        within_circle = np.where(dists <= radius)[0]
        
        if len(within_circle) > 1:
            # Compute the mean position
            mean_lat = np.mean(candidate_points[within_circle][:, 0])
            mean_lng = np.mean(candidate_points[within_circle][:, 1])
            updated_candidates.append(Point(mean_lng, mean_lat))
            used[within_circle] = True
        else:
            updated_candidates.append(Point(center_point[1], center_point[0]))
            used[i] = True
    
    return updated_candidates

def plot_results(df, radius, overlap_centers, valid_candidates, grouped_candidates):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot circles and overlap centers
    ax1.scatter(df['lng'], df['lat'], color='blue', label='Data Points')
    
    # Draw circles around each point
    for i, point in df.iterrows():
        circle = plt.Circle((point['lng'], point['lat']), radius, color='red', fill=False, linestyle='--', linewidth=1.5)
        ax1.add_patch(circle)
    
    # Plot overlap centers
    for center in overlap_centers:
        ax1.scatter(center.x, center.y, color='green', label='Overlap Center', s=100, marker='x')
    
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Circles and Overlap Centers')
    ax1.legend()
    ax1.grid(True)
    
    # Plot raw data points and valid candidates
    ax2.scatter(df['lng'], df['lat'], color='blue', label='Raw Data Points')
    if valid_candidates:
        valid_lngs = [c.x for c in valid_candidates]
        valid_lats = [c.y for c in valid_candidates]
        ax2.scatter(valid_lngs, valid_lats, color='orange', label='Valid Candidates', s=100, marker='x')
    if grouped_candidates:
        grouped_lngs = [c.x for c in grouped_candidates]
        grouped_lats = [c.y for c in grouped_candidates]
        ax2.scatter(grouped_lngs, grouped_lats, color='red', label='Grouped Candidates', s=150, marker='x')
    
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title('Raw Data Points with Valid and Grouped Candidates')
    ax2.legend()
    ax2.grid(True)
    
    plt.show()

# Compute radius and find overlap centers
@app.route('/orchards/<orchard_id>/missing-trees')
def treefinder(orchard_id):
	df=data_pull(orchard_id)
	radius = compute_radius(df)
	overlap_centers = find_overlap_centers(df, radius)
	valid_candidates = evaluate_candidates(overlap_centers, df)
	grouped_candidates = group_candidates_within_circles(valid_candidates, radius)
	missing_trees = [{"lat": point.y, "lng": point.x} for point in grouped_candidates]
	output = jsonify({"missing_trees": missing_trees})
	return output

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

