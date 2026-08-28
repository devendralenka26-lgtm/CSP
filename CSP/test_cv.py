import cv2
import numpy as np
import os
import sys

# Add backend/app to path to allow importing modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from run import CVService, AIService

def run_integration_test():
    print("=== STARTING INTEGRATION TEST ===")
    
    # 1. Create a simulated leaf image for testing
    print("Generating simulated leaf image...")
    # 500x500 dark background image
    test_img = np.zeros((500, 500, 3), dtype=np.uint8)
    # Draw dark background/soil color (brownish-grey)
    test_img[:] = [45, 60, 80]
    
    # Draw a green leaf in the center (ellipse)
    cv2.ellipse(test_img, (250, 250), (180, 100), 45, 0, 360, (35, 180, 55), -1)
    
    # Draw some diseased spots (yellow chlorosis and brown necrosis spots)
    # Spot 1: Yellow
    cv2.circle(test_img, (200, 220), 12, (20, 210, 220), -1)
    # Spot 2: Brown
    cv2.circle(test_img, (280, 240), 18, (15, 60, 110), -1)
    # Spot 3: Yellow
    cv2.circle(test_img, (240, 280), 8, (30, 200, 210), -1)
    # Spot 4: Brown
    cv2.circle(test_img, (180, 250), 10, (10, 45, 95), -1)

    temp_input_path = os.path.join(os.path.dirname(__file__), 'mock_leaf_test.jpg')
    cv2.imwrite(temp_input_path, test_img)
    print(f"Mock leaf saved to {temp_input_path}")

    # 2. Run OpenCV processing pipeline
    uploads_dir = os.path.join(os.path.dirname(__file__), 'backend', 'uploads')
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)

    print("Running leaf through OpenCV segmentation...")
    try:
        cv_results = CVService.process_leaf_image(temp_input_path, uploads_dir)
        print("OpenCV pipeline completed successfully.")
        print(f" - Infection ratio detected: {cv_results['infection_ratio']}%")
        print(f" - Lesion contours found: {cv_results['lesion_count']}")
        print(f" - Outputs created: {cv_results['original_name']}, {cv_results['annotated_name']}, {cv_results['heatmap_name']}")
        
        # Verify files exist
        assert os.path.exists(os.path.join(uploads_dir, cv_results['original_name'])), "Original resized image missing"
        assert os.path.exists(os.path.join(uploads_dir, cv_results['annotated_name'])), "Annotated outline image missing"
        assert os.path.exists(os.path.join(uploads_dir, cv_results['heatmap_name'])), "Heatmap image missing"
        
        # 3. Run predictions through Scikit-Learn Classifiers
        print("Initializing AI ML classifiers...")
        ai_service = AIService()
        
        print("Testing Tomato classification...")
        prediction = ai_service.predict_disease("Tomato", cv_results['infection_ratio'], cv_results['lesion_count'])
        print(f" - Classification: {prediction['disease_name']}")
        print(f" - Severity: {prediction['severity']}")
        print(f" - Confidence: {prediction['confidence']}%")
        print(f" - Organic remedy recommendation: {prediction['treatment_organic']}")
        
        # Assertions
        assert prediction['severity'] in ['Low', 'Medium', 'High'], "Severity categorization failed"
        assert prediction['confidence'] > 50.0, "Confidence score simulation failed"

        print("=== TEST COMPLETED SUCCESSFULLY ===")
        
        # Cleanup mock test input file
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
            
        return True
    except Exception as e:
        print(f"!!! INTEGRATION TEST FAILED: {e} !!!", file=sys.stderr)
        return False

if __name__ == '__main__':
    success = run_integration_test()
    sys.exit(0 if success else 1)
