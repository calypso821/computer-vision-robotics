



kernel = [0.5, 1, 0.3]
signal = [0, 1, 1, 1, 0, 0.7, 0.5, 0.2, 0, 0, 1, 0]

def apply_filter(kernel, signal):
    new_signal = []
    index = 0
    while index < len(signal) - 2:
        dot = signal[index] * kernel[0] + signal[index + 1] * kernel[1] + signal[index + 2] * kernel[2]
        new_signal.append(dot)
        index += 1
    return new_signal



new_signal = apply_filter(kernel, signal)
print(new_signal)