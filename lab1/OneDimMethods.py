def double_scale_method(f, a, b, epsilon):
    delta = epsilon / 100
    c = 0
    while(b - a > epsilon):
        c = (b + a) / 2
        x1 = c - delta 
        x2 = c + delta
        f_x1 = f(x1)
        f_x2 = f(x2)
        print(a, x1, x2, b)
        if(f_x1 < f_x2):
            b = x2
        else: 
            a = x1
    print(a, b)
    return f((b+a)/2)
    
    

def cut_section_procedure(a, x1, x2, b, f_x1, f_x2):
    if f_x1 <= f_x2:
        return a, x2, x1, f_x1
    else:
        return x1, b, x2, f_x2



def golden_scale_method(f, a, b, epsilon):
    g_c = 1.618033988
    x, f_x = 0.0, 0.0
    x2 = (b - a)/g_c + a 
    x1 = b - (b - a)/(g_c)
    f_x1, f_x2 = f(x1), f(x2)
    print(a, x1, x2, b)
    a, b, x, f_x = cut_section_procedure(a, x1, x2, b, f_x1, f_x2)
    while(b - a > epsilon):
        y = a + b - x
        if x < y:
            x1, x2 = x, y
            f_x1, f_x2 = f_x, f(y)
        else: 
            x1, x2 = y, x
            f_x1, f_x2 = f(y), f_x
        print(a, x1, x2, b)
        a, b, x, f_x = cut_section_procedure(a, x1, x2, b, f_x1, f_x2) 
    print(a, x1, x2, b)
    return f((a + b)/2)


def getFibbonachies(epsilon):
    fibbs = [] 
    fibbs.append(1)
    fibbs.append(1)
    while(fibbs[len(fibbs) - 1] < 1/epsilon):
        f2 = fibbs[len(fibbs) - 1]
        f1 = fibbs[len(fibbs) - 2]
        fibbs.append(f1 + f2)
    return fibbs
    


def fibbonaci_method(f, a, b, epsilon):
    F = getFibbonachies(epsilon)
    N = len(F) - 2
    l = b - a
    delta = epsilon / 100
    x2 = a + F[N]/F[N+1] * l + (-1)**(N+1)*2*delta/F[N+1]
    x1 = a + F[N-1]/F[N+1] * l
    f_x1, f_x2 = f(x1), f(x2)
    print(a, x1, x2, b)
    a, b, x, f_x = cut_section_procedure(a, x1, x2, b, f_x1, f_x2)
    for i in range(2, N+1):
        y = a + F[N-i]/F[N+1] * l + (-1)**(N+1)*2*delta/F[N+1]
        if x == y: y = b - F[N-i]/F[N+1] * l + (-1)**(N+1)*2*delta/F[N+1]
        if x < y:
            x1, x2 = x, y
            f_x1, f_x2 = f_x, f(y)
        else: 
            x1, x2 = y, x
            f_x1, f_x2 = f(y), f_x
        print(a, x1, x2, b)
        a, b, x, f_x = cut_section_procedure(a, x1, x2, b, f_x1, f_x2) 
    print(a, b)
    return f_x


def find_parabola_minimum(x1, x2, x3, f1, f2, f3):
    a = f1/((x1-x2)*(x1-x3)) + f2/((x2-x1)*(x2-x3)) + f3/((x3-x1)*(x3-x2))
    b = (f1 * (x2 + x3))/((x1-x2)*(x3-x1)) + (f2 * (x1 + x3))/((x2-x1)*(x3-x2)) + (f3 * (x1 + x2))/((x3-x1)*(x2-x3))
    return -b/(2*a)


def quadratic_approx(f, a, b, epsilon):
    x1, x3 = a, b
#     x2 = a + (b - a) * 1/1.618033988
    x2 = 3
    f1, f2, f3 = f(x1), f(x2), f(x3)
    print(x1, x2, x3)
    while(True):
        xm = find_parabola_minimum(x1, x2, x3, f1, f2, f3)
        fm = f(xm)
        if xm < x3 and xm > x2 and fm <= f2:
            x1, x2 = x2, xm
            f1, f2 = f2, fm
        elif xm < x3 and xm > x2 and fm > f2:
            x3 = xm
            f3 = fm
        elif xm < x2 and xm > x1 and fm <= f2:
            x2, x3 = xm, x2
            f2, f3 = fm, f2
        elif xm < x2 and xm > x1 and fm > f2:
            x1 = xm
            f1 = fm
        print(xm, fm)
        if(x3-x2 < epsilon or x2-x1 < epsilon): break
    return fm