from Clough_Toucher_derivation import *
import pretty_print as pp

file = open("clough_tocher_autogen_constraint_matrices.hpp", "w")

file.write("#pragma once\n\n")

polys_C1_f = derive_C1_polynomials(vtx_unknowns,side_deriv_unknowns,mid_deriv_unknowns)
runTests(polys_C1_f)
CT_matrices = compute_CT_matrices(polys_C1_f)

L_d2L = generate_L2L(polys_C1_f, node_bary, node_subtri, all_unknowns)
L_d2L_ind = L_d2L[0:12,0:12]
L_d2L_dep = L_d2L[12:19,0:12]
L_L2d_ind = L_d2L_ind.inverse_GE()
L_ind2dep =  L_d2L_dep*L_L2d_ind
c_e = generate_ce(polys_C1_f)

test_Lagrange_consistency(polys_C1_f, L_L2d_ind, L_ind2dep)

# L_L2d_ind
file.write("inline void L_L2d_ind_matrix(double L[12][12]){\n")
s = pp.C99_print_tensor(L_L2d_ind, "L")
ss = s.replace(", ", "][")
file.write(ss)
file.write("}\n\n")

# L_d2L_dep
file.write("inline void L_d2L_dep_matrix(double L[7][12]){\n")
s = pp.C99_print_tensor(L_d2L_dep, "L")
ss = s.replace(", ", "][")
file.write(ss)
file.write("}\n\n")

# c_e
file.write("inline void c_e_matrix(double c[5]){\n")
for i in range(len(c_e)):
    file.write("c[{}] = {};\n".format(i, c_e[i]))
file.write("}\n\n")

file.close()