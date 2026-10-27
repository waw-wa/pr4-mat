import os
import random
import math
import matplotlib.pyplot as plt

# Фіксація генератора псевдовипадкових чисел для відтворюваності
random.seed(993)

# Створення папки для графіків
os.makedirs("graphics", exist_ok=True)

#========================================================
#Блок 1. Моделювання байєсівського оновлення ймовірностей
#========================================================
def run_block_1():
    print("=== БЛОК 1: Моделювання байєсівського оновлення ймовірностей ===")
    
    # Апріорні ймовірності гіпотез
    p_H1 = 0.90  # Легітимний
    p_H2 = 0.08  # Бот
    p_H3 = 0.02  # Атака
    
    # Умовні ймовірності події А
    p_A_H1 = 0.01
    p_A_H2 = 0.25
    p_A_H3 = 0.90
    
    N_list = [100, 1000, 10000, 100000, 1000000]
    results_W = []
    
    theor_P_A = p_H1*p_A_H1 + p_H2*p_A_H2 + p_H3*p_A_H3
    theor_P_H3_A = (p_H3 * p_A_H3) / theor_P_A
    
    for N in N_list:
        count_A = 0
        count_H3_and_A = 0
        
        for _ in range(N):
            r_H = random.random()
            if r_H < p_H1:
                H = 1
                p_A = p_A_H1
            elif r_H < p_H1 + p_H2:
                H = 2
                p_A = p_A_H2
            else:
                H = 3
                p_A = p_A_H3
                
            if random.random() < p_A:
                count_A += 1
                if H == 3:
                    count_H3_and_A += 1
                    
        W_A = count_A / N if N > 0 else 0
        W_H3_A = count_H3_and_A / count_A if count_A > 0 else 0
        results_W.append(W_H3_A)
        
        print(f"N={N:<8} | W(A)={W_A:.4f} | W(H3|A)={W_H3_A:.4f} | Похибка={abs(W_H3_A - theor_P_H3_A):.5f}")

    # Побудова графіка
    plt.figure(figsize=(10, 6))
    plt.plot(N_list, results_W, marker='o', label="Оцінка W(H3|A)", color='blue')
    plt.axhline(y=theor_P_H3_A, color='red', linestyle='--', label=f"Теорія P(H3|A) = {theor_P_H3_A:.4f}")
    plt.xscale('log')
    plt.xlabel("Кількість запитів N (log scale)")
    plt.ylabel("Апостеріорна ймовірність")
    plt.title("Блок 1: Збіжність апостеріорної оцінки W(H3|A)")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.savefig("graphics/block1_bayesian_update.png", dpi=300)
    plt.close()
    print("-> Графік збережено в graphics/block1_bayesian_update.png\n")

#========================================================
#Блок 2,Варіант 2. Точність аналізатора коду SAST (Static Analysis Tool) 
#========================================================
def run_block_2():
    print("=== БЛОК 2: Точність аналізатора коду SAST (Варіант 2) ===")
    N = 100000
    p_H1 = 0.05  # Дефект
    p_H2 = 0.95  # Чистий код
    p_A_H1 = 0.90 # True Positive
    p_A_H2 = 0.02 # False Positive
    
    count_A = 0
    count_H1_and_A = 0
    
    for _ in range(N):
        r_H = random.random()
        is_H1 = r_H < p_H1
        p_A = p_A_H1 if is_H1 else p_A_H2
        
        if random.random() < p_A:
            count_A += 1
            if is_H1:
                count_H1_and_A += 1
                
    W_A = count_A / N
    W_H1_A = count_H1_and_A / count_A if count_A > 0 else 0
    theor_P_A = p_H1*p_A_H1 + p_H2*p_A_H2
    theor_P_H1_A = (p_H1 * p_A_H1) / theor_P_A
    
    print(f"Загальна ймовірність спрацьовування: W(A)={W_A:.4f} (Теорія: {theor_P_A:.4f})")
    print(f"Точність (Precision): W(H1|A)={W_H1_A:.4f} (Теорія: {theor_P_H1_A:.4f})")
    print(f"Абсолютна похибка W(H1|A): {abs(W_H1_A - theor_P_H1_A):.5f}\n")

#========================================================
#Блок 3. Наївний байєсівський класифікатор (Naive Bayes)
#========================================================
def run_block_3():
    print("=== БЛОК 3: Наївний байєсівський класифікатор (Spam/Ham) ===")
    N = 10000
    p_spam, p_ham = 0.30, 0.70
    p_w1_spam, p_w2_spam = 0.80, 0.60
    p_w1_ham, p_w2_ham = 0.05, 0.10
    
    TP = TN = FP = FN = 0
    
    for _ in range(N):
        is_spam_true = random.random() < p_spam
        
        # Генерація ознак
        w1 = 1 if random.random() < (p_w1_spam if is_spam_true else p_w1_ham) else 0
        w2 = 1 if random.random() < (p_w2_spam if is_spam_true else p_w2_ham) else 0
        
        # Розрахунок MAP
        spam_score = p_spam * (p_w1_spam if w1 else 1-p_w1_spam) * (p_w2_spam if w2 else 1-p_w2_spam)
        ham_score = p_ham * (p_w1_ham if w1 else 1-p_w1_ham) * (p_w2_ham if w2 else 1-p_w2_ham)
        
        pred_spam = spam_score > ham_score
        
        if is_spam_true and pred_spam: TP += 1
        elif not is_spam_true and not pred_spam: TN += 1
        elif not is_spam_true and pred_spam: FP += 1
        elif is_spam_true and not pred_spam: FN += 1
        
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Confusion Matrix: TP={TP}, FP={FP}, TN={TN}, FN={FN}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}\n")

if __name__ == "__main__":
    run_block_1()
    run_block_2()
    run_block_3()