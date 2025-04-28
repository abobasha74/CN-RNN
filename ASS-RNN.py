import numpy as np

np.random.seed(42)

# الكلمات اللي هنشتغل عليها
text = ["the", "cat", "is", "fluffy"]
vocab = {word: i for i, word in enumerate(text)}
vocab_size = len(vocab)
hidden_size = 3

# بنجهز مصفوفة فيها الكلمات one-hot
X = np.zeros((len(text), vocab_size))
for t, word in enumerate(text):
    X[t, vocab[word]] = 1.0

# الهدف بتاعنا اللي عايزين نتعلم عليه
y_target = np.zeros(vocab_size)
y_target[vocab["fluffy"]] = 1.0

# بنحط أوزان الشبكة بشكل عشوائي
Wx = np.random.randn(hidden_size, vocab_size) * 0.1
Wh = np.random.randn(hidden_size, hidden_size) * 0.1
Wy = np.random.randn(vocab_size, hidden_size) * 0.1

learning_rate = 0.01
num_iterations = 1000

# دوال التنشيط
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x) ** 2

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# نبدأ التدريب
for iteration in range(num_iterations):
    h = [np.zeros(hidden_size)]
    a = []
    y_pred = []
    
    for t in range(len(text)):
        at = Wx @ X[t] + Wh @ h[t]
        ht = tanh(at)
        yt = softmax(Wy @ ht)
        
        h.append(ht)
        a.append(at)
        y_pred.append(yt)
    
    # حساب الخسارة
    loss = np.sum((y_pred[3] - y_target) ** 2)
    
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    dWy = np.zeros_like(Wy)
    
    dy3 = 2 * (y_pred[3] - y_target)
    
    softmax_deriv = y_pred[3] * (1 - y_pred[3])
    dh3 = Wy.T @ (dy3 * softmax_deriv)
    dWy += np.outer(dy3 * softmax_deriv, h[3])
    
    dh = dh3
    for t in range(3, 0, -1):
        da = tanh_derivative(a[t]) * dh
        
        dWh += np.outer(da, h[t-1])
        dWx += np.outer(da, X[t-1])
        
        dh = Wh.T @ da
    
    # تحديث الأوزان
    Wx -= learning_rate * dWx
    Wh -= learning_rate * dWh
    Wy -= learning_rate * dWy
    
    # كل شوية نطبع اللوس عشان نشوف الدنيا ماشية ازاي
    if iteration % 100 == 0:
        print(f"Iteration {iteration}, Loss: {loss:.4f}")

# دلوقتي نجرب الشبكة بعد التدريب
h = np.zeros(hidden_size)
for t in range(3):
    a = Wx @ X[t] + Wh @ h
    h = tanh(a)

y_final = softmax(Wy @ h)
predicted_word_idx = np.argmax(y_final)
predicted_word = list(vocab.keys())[predicted_word_idx]

print(f"\nPredicted fourth word: {predicted_word}")
print(f"Prediction probabilities: {y_final}")
