# adv_ML_LogisticRegression

FUNCTION LogisticRegression(X, y, learning_rate, num_iterations):
    // Inisialisasi parameter
    theta = ZEROS(number_of_features + 1)
    m = number_of_samples_in_X
    
    // Tambahkan kolom bias ke X
    X = ADD_COLUMN_OF_ONES(X)
    
    FOR iteration = 1 TO num_iterations:
        // Forward pass
        z = X * theta
        h = SIGMOID(z)
        
        // Compute gradient
        gradient = (1/m) * X^T * (h - y)
        
        // Update parameters
        theta = theta - learning_rate * gradient
        
        // Optional: Compute and store cost
        cost = -(1/m) * SUM(y * LOG(h) + (1-y) * LOG(1-h))
        STORE(cost)
    
    RETURN theta

FUNCTION Predict(X, theta):
    // Tambahkan kolom bias ke X
    X = ADD_COLUMN_OF_ONES(X)
    
    // Compute probabilitas
    z = X * theta
    probabilities = SIGMOID(z)
    
    // Klasifikasi (threshold 0.5)
    predictions = ROUND(probabilities)
    
    RETURN predictions

FUNCTION SIGMOID(z):
    RETURN 1 / (1 + e^(-z))
