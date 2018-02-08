
# from scipy.optimize import minimize
#
# #testing list of funcitons
#
# def sample(i, l):
#     return i*l
#
# class Samp():
#     def __init__(self, i):
#         self.i = i
#     def __call__(self, l):
#         return l*self.i
#
# a = [Samp(i) for i in range(10)]
#
# for e in a:
#     print(e(3))
#
#
# def objective(x):
#     return x**2-10
#
# def cons1(x):
#     return x-10
#
# def cons2(x):
#     return 1-x
#
# con1 = {'type': 'ineq', 'fun': cons1}
# con2 = {'type': 'ineq', 'fun': cons2}
# cons = [con1, con2]
# sol = minimize(objective, [2.], method = 'SLSQP', bounds = [(0, 10)], constraints = cons, options={'disp': True})
# print(sol.x)
