import numpy as np

# must be either 16 or 32 or 64
N_BITS = 16

types = {
    16: [np.float16, np.uint16],
    32: [np.float32, np.uint32],
    64: [np.float64, np.uint64],
}

def binary(num):
    t = types[N_BITS]
    uints = np.asarray(num, dtype=t[0]).view(t[1]).item()
    return bin(uints)[2:]

def from_binary(bit):
    t = types[N_BITS]
    uints = int(bit, 2)
    return np.asarray(uints, dtype=t[1]).view(t[0]).item()

def get_linspaced_population(a, b, n=50):
    nums = np.linspace(a, b, n)
    bit_vectors =np.vectorize(binary)(nums)
    return list(map(lambda x: x.rjust(N_BITS, '0'), bit_vectors))

def extimate_population_quality(population, f):
    f_v = np.vectorize(f)
    fb_v = np.vectorize(from_binary)
    return f_v(fb_v(population))

def canonical_genetical_algorithm(f):
    pass

if __name__ == "__main__":
    population = get_linspaced_population(-10, 10, 20)
    # print("The population:")
    # for i in population:
    #     print(f"{i}    \t{from_binary(i)}")

    # print(''.rjust(20, '#'))

    print("the quality:")
    qualities = extimate_population_quality(population, lambda x: np.power(x - 1, 2))
    for i in range(len(population)):
        print(f"{population[i]}    \t{qualities[i]}")


