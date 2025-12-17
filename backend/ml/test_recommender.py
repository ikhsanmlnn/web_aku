# test_recommender.py
from course_recommender import CourseRecommenderDicoding

# Initialize
recommender = CourseRecommenderDicoding()

# Test 1: Load data
print("=" * 60)
print("TEST 1: Loading Data")
print("=" * 60)
recommender.load_data_from_api()

# Test 2: Check data structure
print("\n" + "=" * 60)
print("TEST 2: Data Structure")
print("=" * 60)
if recommender.courses_data is not None:
    print(f"Total courses: {len(recommender.courses_data)}")
    print(f"Columns: {recommender.courses_data.columns.tolist()}")
    
    # ✅ FIX: Gunakan kolom yang benar
    if 'course_level' in recommender.courses_data.columns:
        display_cols = ['name', 'course_level', 'learning_path_name']
    else:
        display_cols = ['name', 'course_level_standard', 'learning_path_name']
    
    print(f"\nSample data:")
    print(recommender.courses_data.head(3)[display_cols].to_string())

# Test 3: Get recommendations
print("\n" + "=" * 60)
print("TEST 3: Recommendations")
print("=" * 60)
results = recommender.recommend("DevOps Engineer", user_level="Beginner", top_k=5)
print(f"\nFound {len(results)} recommendations:")
print(results.to_string())