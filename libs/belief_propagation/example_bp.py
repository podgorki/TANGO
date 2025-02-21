# %%
from factor_graph import *

print("Running BP example...")
mrf = string2factor_graph('f1(a,b)f2(b,c,d)f3(c)')

f1 = factor(['a', 'b'],      np.array([[2,3],[6,4]]))
f2 = factor(['b', 'd', 'c'], np.array([[[7,2,3],[1,5,2]],[[8,3,9],[6,4,2]]]))
f3 = factor(['c'],           np.array([5, 1, 9]))

mrf.change_factor_distribution('f1', f1)
mrf.change_factor_distribution('f2', f2)
mrf.change_factor_distribution('f3', f3)

bp = belief_propagation(mrf)
dist = bp.belief('b').get_distribution()

print(f"Expected: array([0.37398374, 0.62601626])")
print(f"Actual: {dist}")


print("Running Loopy BP example...")
mrf = string2factor_graph('f1(a,b)f2(a,c)f3(b,c)')
f1 = factor(['a', 'b'],  np.array([[2,3],[6,4]]))
f2 = factor(['a', 'c'],  np.array([[7,2,3],[1,5,2]]))
f3 = factor(['b', 'c'],  np.array([[7,9,3],[6,4,2]]))

mrf.change_factor_distribution('f1', f1)
mrf.change_factor_distribution('f2', f2)
mrf.change_factor_distribution('f3', f3)

print("Compute exact marginal distribution of b and normalize it")
exact = factor_marginalization(joint_distribution([f1, f2, f3]), ['a', 'c']).get_distribution()
exact = exact / np.sum(exact)

print("Expected: array([0.63451777, 0.36548223])")
print(f"Actual: {exact}")

print("run the loopy belief propagation algorithm")
lbp = loopy_belief_propagation(mrf)
tol = []

for i in range(15):
    tol.append(np.linalg.norm(lbp.belief('b', i).get_distribution() - exact))

plt.figure(figsize=(8,6))
plt.semilogy(tol, label=r"$\|b_k - b^*\|_2$", color='navy')
plt.xlabel(r"Number of iteration, $k$", fontsize=15)
plt.ylabel(r"Convergence rate", fontsize=15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend(loc="best", fontsize=15)
plt.plot(tol)
# %%
