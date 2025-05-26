import numpy as np
import matplotlib.pyplot as plt

# 1. Задаём параметры задачи
R = 0.5               # радиус кривизны линзы, м
lambda0 = 550e-9      # центральная длина волны для монохроматического случая, м
delta_lambda = 50e-9  # ширина спектра для квазимонохром. случая, м
wavelengths = np.linspace(lambda0 - delta_lambda/2,
                          lambda0 + delta_lambda/2, 11)

# 2. Определяем пространство моделирования
size = 5e-3   # половина ширины области по x и y, м (итого 10 мм × 10 мм)
N = 1000      # число точек по каждой оси
x = np.linspace(-size, size, N)
y = x.copy()
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2)        # расстояние от центра
t = r**2 / (2 * R)              # толщина воздушной плёнки

# 3. Монохроматический случай (λ = 550 нм)
phi_mono = 2 * np.pi * 2 * t / lambda0 + np.pi   # разность фаз (с т.з. π-сдвига на отражении)
I_mono = np.cos(phi_mono / 2)**2                 # интенсивность ∝ cos²(Δφ/2)
# нормируем для отображения
I_mono = (I_mono - I_mono.min()) / (I_mono.max() - I_mono.min())

# 3.1. Рисуем чёрно-белое изображение
plt.figure(figsize=(5,5))
plt.imshow(I_mono, extent=(-size*1e3, size*1e3, -size*1e3, size*1e3),
           cmap='gray')
plt.xlabel('x, мм')
plt.ylabel('y, мм')
plt.title('Кольца Ньютона (монохроматическая λ = 550 нм)')
plt.tight_layout()
plt.show()

# 3.2. Радиальный профиль вдоль оси x
r_line = x.copy()
t_line = r_line**2 / (2 * R)
phi_line = 2 * np.pi * 2 * t_line / lambda0 + np.pi
I_line_mono = np.cos(phi_line / 2)**2

plt.figure(figsize=(6,3))
plt.plot(r_line*1e3, I_line_mono)
plt.xlabel('r, мм')
plt.ylabel('Интенсивность')
plt.title('Профиль интенсивности (монохро.)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# 4. Квазимонохроматический случай: смешение 11 длин волн
def wavelength_to_rgb(wl, gamma=0.8):
    """Преобразует длину волны wl (в метрах) в RGB-приближение."""
    wl_nm = wl * 1e9
    # базовая спектральная цветопередача
    if 380 <= wl_nm <= 440:
        r = -(wl_nm - 440) / 60; g = 0; b = 1
    elif 440 < wl_nm <= 490:
        r = 0; g = (wl_nm - 440) / 50; b = 1
    elif 490 < wl_nm <= 510:
        r = 0; g = 1; b = -(wl_nm - 510) / 20
    elif 510 < wl_nm <= 580:
        r = (wl_nm - 510) / 70; g = 1; b = 0
    elif 580 < wl_nm <= 645:
        r = 1; g = -(wl_nm - 645) / 65; b = 0
    elif 645 < wl_nm <= 780:
        r = 1; g = 0; b = 0
    else:
        r = g = b = 0

    # учёт восприятия на краях видимого диапазона
    if wl_nm < 420:
        factor = 0.3 + 0.7*(wl_nm-380)/40
    elif wl_nm > 700:
        factor = 0.3 + 0.7*(780-wl_nm)/80
    else:
        factor = 1

    return np.array([r, g, b]) * factor**gamma

# для каждой длины волны считаем интенсивность и окрашиваем
stack = []
for wl in wavelengths:
    phi = 2 * np.pi * 2 * t / wl + np.pi
    I = np.cos(phi/2)**2
    rgb = wavelength_to_rgb(wl)
    stack.append(I[:, :, None] * rgb[None, None, :])

# суммируем вклад всех длин волн и нормируем
I_color = np.sum(stack, axis=0)
I_color /= I_color.max()

# 4.1. Рисуем цветное изображение
plt.figure(figsize=(5,5))
plt.imshow(I_color, extent=(-size*1e3, size*1e3, -size*1e3, size*1e3))
plt.xlabel('x, мм')
plt.ylabel('y, мм')
plt.title('Кольца Ньютона (квазимонохр. 550±25 нм)')
plt.tight_layout()
plt.show()

# 4.2. Радиальный профиль цветной картины
center = N // 2
r_vals = r[center, :]*1e3  # в мм
I_profile_color = I_color[center, :].mean(axis=1) if I_color.ndim==3 else I_color[center, :]

plt.figure(figsize=(6,3))
plt.plot(r_vals, I_profile_color)
plt.xlabel('r, мм')
plt.ylabel('Интенсивность (усреднённая)')
plt.title('Профиль интенсивности (квазимонохр.)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
