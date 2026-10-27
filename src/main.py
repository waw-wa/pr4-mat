import os
import random
import math
import matplotlib.pyplot as plt

# Фіксація генератора псевдовипадкових чисел для відтворюваності
random.seed(993)

# Створення папки для графіків
os.makedirs("graphics", exist_ok=True)

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

if __name__ == "__main__":
    run_block_1()