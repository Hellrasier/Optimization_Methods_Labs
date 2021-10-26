def double_scale_method(f, a, b, epsilon):
    delta = epsilon / 5
    c = 0
    while(b - a > epsilon):
        c = (b + a) / 2
        x1 = c - delta
        x2 = c + delta
        f_x1 = f(x1)
        f_x2 = f(x2)
        print(a, b, f(c))
        if(f_x1 < f_x2):
            a = x1
        else:
            b = x2
    return f(c)
