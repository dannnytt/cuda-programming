import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data/results.csv")

# Строим график
plt.figure(figsize=(20, 12))
plt.plot(data["Threads"], data["Time(ms)"], marker='o', linestyle='-')

# Настройки графика
plt.xscale("log")   
plt.xlabel("Количество потоков")
plt.ylabel("Время выполнения (мс)")
plt.title("Зависимость времени выполнения от количества потоков")
plt.grid(True)

# Устанавливаем ось X с реальными числами вместо степеней 10
plt.xticks(data["Threads"], labels=data["Threads"])

# Сохранение и показ графика
plt.savefig("data/output.png")
plt.show()
