from sympy import Rational, Matrix, symbols, det, collect, simplify, solve, diff, Eq, expand, Poly
import sympy as sp
import numpy as np

#  *********  helper functions *********
def nxt(i):
    return (i+1) % 3
def prev(i):
    return (i+2) % 3
    
# Equation of a line in homogeneous coordinates
def ep2line(a, b):
    """Returns the implicit form of a line passing through points a and b."""
    return collect(det(Matrix([a.T, b.T, Matrix([[u, v, w]])])), [u, v, w])

def ep2param_line(a, b):
    """Returns the param equation of a line passing through points a and b."""
    return a + (b - a) * t

# Derivatives in homogeneous coordinates
# these are linearly dependent, using 3 for symmetry for variables u,v,w
def gradb(f):
    return [
        (1 / Rational(2)) * (2 * diff(f, u) - diff(f, v) - diff(f, w)),
        (1 / Rational(2)) * (2 * diff(f, v) - diff(f, u) - diff(f, w)),
        (1 / Rational(2)) * (2 * diff(f, w) - diff(f, v) - diff(f, u))
    ]



#  ********* global definitions, barycentric coord symbols *********

u, v, w = symbols('u v w')
uvwvars = (u,v,w)

# parametric line parameter
t = symbols('t')

#  ********* global definitions, triangle geometry *********

# reference triangle geometry
# triangle corners, homogeneous coords
pt = [
        Matrix([Rational(1), Rational(0), Rational(0)]),
        Matrix([Rational(0), Rational(1), Rational(0)]),
        Matrix([Rational(0), Rational(0), Rational(1)])
    ]
 # edge midpoints
midpt = [
        (pt[0] + pt[1]) / 2,
        (pt[1] + pt[2]) / 2,
        (pt[2] + pt[0]) / 2
    ]
    
center = (pt[0] + pt[1] + pt[2]) / 3
    
# three subtriangles
CTtri = [
        [pt[0], pt[1], center],  # adjacent to the edge w = 0, 01
        [pt[1], pt[2], center],  # adjacent to the edge u = 0, 12
        [pt[2], pt[0], center]   # adjacent to the edge v = 0, 20
    ]

# Lines passing through the center and triangle vertices
lineseq_center  = [simplify(ep2line(pt[i], center)) for i in range(3)]

# equations of lines bounding subtriangles
CTtri_bounds = [ Matrix([ [q.coeff(bc) for bc in uvwvars] for q in p]) for p in list( map( lambda x: [ep2line(x[0],x[1]), 
                                     ep2line(x[1],x[2]), ep2line(x[2],x[0])], CTtri))]


#  ********* global definitions, Lagrange nodes *********

#subtriangle corner barycentric coordinates
bp0 = np.array([Rational(1),Rational(0),Rational(0)])
bp1 = np.array([Rational(0),Rational(1),Rational(0)])
bp2 = np.array([Rational(0),Rational(0),Rational(1)])
bc  = np.array([Rational(1,3),Rational(1,3),Rational(1,3)])

# barycentric coords of all Lagrange nodes, multiplied by 9, so that these are all integers
# computed as linear combinations of corners of subtriangles
# coefficients should sum up to 9
node_bary = [
  9*bp0, 9*bp1, 9*bp2, # corners
  6*bp0+3*bp1, 3*bp0+6*bp1,   6*bp1+3*bp2, 3*bp1+6*bp2,   6*bp2+3*bp0, 3*bp2+6*bp0, # 2 nodes  per exterior edge,split the edges 2:1 and 1:2
  3*bp0+3*bp1+3*bc, 3*bp1+3*bp2+3*bc, 3*bp2+3*bp0+3*bc, # subtriangle center nodes  
  6*bp0+3*bc, 3*bp0+6*bc,   6*bp1+3*bc, 3*bp1+6*bc,   6*bp2+3*bc, 3*bp2+6*bc, # 2 nodes  per interior edge split the edges 2:1 and 1:2
  9*bc # center node 
]  
subtri_nodes = [ 
    [0,1, 3+0, 3+1,  9 ,  12+0, 12+1, 12+2, 12+3,  18],
    [1,2, 3+2, 3+3,  10,  12+2, 12+3, 12+4, 12+5,  18],
    [2,0, 3+4, 3+5,  11,  12+4, 12+5, 12+0, 12+1,  18],
]

# a helpde function to create symbolic names for nodes based on their barycentric coords
def bary2name(bc):
    return 'c'+''.join([ str(c) for c in bc])

# dictionary  barycentric coordinate tuples -> symbolic name for an node
node_vars_dict = {tuple(map(lambda x: int(x), b)): symbols(bary2name(b)) for b in node_bary}

# array of barycentric coord tuples in the same order as in node_bary 
ind2bary = [tuple(map(lambda x: int(x), b))  for b in node_bary]

# array of variables in the order matching node_bary
node_vars = [ node_vars_dict[ind2bary[i]] for i in range(19)]

# polynomials that can be used to evaluate each of the Lagrange nodes
node_subtri = \
[0, 1, 2,  # corners, one index of two possible 
 0, 0, 1, 1, 2,2, # exterior edges 
 0, 1, 2, # subtriangle centers 
 0, 0, 1, 1, 2, 2, # interior edges, one index of two possible 
 0 # any polynomial can be used
]


# ********* global definitions, boundary dofs of the element *********

# boundary data: corners, derivatives along edges, midedge derivatives in the opposite vertex direction
p0,p1,p2 = symbols(['p0','p1','p2'])
vtx_unknowns = (p0,p1,p2)
G01,G10,G12,G21,G20,G02 =  symbols('G01 G10 G12 G21 G20 G02')
side_deriv_unknowns =(G01,G10,G12,G21,G20,G02)
N01, N12, N20 = symbols('N01 N12 N20')
mid_deriv_unknowns = (N01, N12, N20)
all_unknowns = vtx_unknowns + side_deriv_unknowns + mid_deriv_unknowns 

# ********* main function: derives subtriangle polynomials *********

def derive_C1_polynomials(vtx_unknowns,side_deriv_unknowns,mid_deriv_unknowns):
    """ This function derives expressions for Clough-Tocher polynomials on subtriangles  
    with coefficients determined by the boundary data and joined with C1 continuity 
    in the interior.  The boundary data includes: 
    -- vtx-unknowns:  function vertex values 
    -- side_deriv_unknowns: values of scaled derivatives Gij along triangle sides at each vertex
       i.e.  grad F(p_i) dot e_ij where p_i is the vertex and e_ij is the edge vector
    -- mid_deriv_unknowns at the midpoints of scaled derivatives along vectors to vertices 
    i.e., grad F(m_ij) dot (p_k - m_ij), where k is the vertex opposite ij, m_ij = (p_i + p_j)/2
    The ordering of the unknowns is   p0,p1,p2, G01,G10, G12,G21, G20, G21, N01, N12, N20
    
    The function returns a list of 3 polynomials, one per triangle, in variables u,v,w
    and with coefficienets dependent only on the unknowns passed in
    """
    global pt,  midpt, center, CTtri, lineseq_center

    # coefficients of an inknown polynomial for the first triangle, to be sovled for
    q300, q030, q210, q120, q201, q021, q003, q111, q102, q012 = \
        symbols('q300 q030 q210 q120 q201 q021 q003 q111 q102 q012')
    first_poly_coeffs = (q300, q030, q210, q120, q201, q021, q003, q111, q102, q012)

    # additional coefficients to define other polynomials relative to the first one
    lambda_ = symbols('lambda_0:4')
    mu = symbols('mu_0:4')
    nu = symbols('nu_0:4')
    
    
    # Polynomial for subtriangle 0
    startpoly = q300 * u**3 + q030 * v**3 + q210 * u**2 * v + q120 * u * v**2 \
                + q201 * u**2 * w + q021 * v**2 * w + q003 * w**3 + q111 * u * v * w \
                + q102 * u * w**2 + q012 * v * w**2
    
    # Define 3 polynomials joined with C1 continuity walking around the center vertex and adding a term of the form 
    # d = (lambda[i]*u + mu[i]*v + nu[i]*w)*(L)^2 each time we cross a line with equation L(u,v,w)=0
    # this ensures C1: d(u,v) has zero values and derivatives on L, so the two polys on two sides 
    # differ by d(u,v) only.  Crossing the last line takes us back to the original triangle, 
    # so the polynomial should be the same, which allows to eliminate all unknowns. 
    
    polys = [
        startpoly + sum( (lambda_[i]*u + mu[i] * v + nu[i] * w) * lineseq_center[nxt(i)]**2 for i in range(j))
        for j in range(4) ]
    
    # solve for all mu's and nu's and one of lambdas 
    # based on equating the last polynimal and first
    depvars = solve( Poly(polys[3]-polys[0], [u,v,w]).coeffs(), 
          [lambda_[0],mu[0],mu[1],mu[2], nu[0], nu[1], nu[2]])
    
    indepvars = first_poly_coeffs + (lambda_[1], lambda_[2])
    
    # Substitute into the C1 polynomials
    polys_C1 = [poly.subs(depvars) for poly in polys]
    
    # Values at triangle corners
    vp = [polys_C1[0].subs({u: 1, v: 0, w: 0}),
         polys_C1[1].subs({u: 0, v: 1, w: 0}),
         polys_C1[2].subs({u: 0, v: 0, w: 1})
        ]
    
    # Variables for side derivatives at vertices
    
    # variable permutations for 6 scaled derivatives along triangle sides, 
    # eg at corner point #0, with barycentric coords [1,0,0], i.e. u=1, and with exterior edge corresponding to w = 0, i.e.,
    # the increasing variable along the edge is v, so we set side_vars[0][0] (in this case w) to 0, differentiate with respect to side_vars[0][1] (in this case v) 
    # and  then set side_vars[[0][1] = 0 and side_vars[0][2] (in this case u)  to 1. 
    # side_vars[i][0]: side line equation is side_vars[i][0] == 0
    # side_vars[i][1]: variable changing along the edge line, with value 0 at the vertex i
    # side_vars[i][2]: variable changing along the edge line, with value 1 at the vertex i
    
    #side_vars := [ [w,v,u],[w,u,v],[u,w,v],[u,v,w],[v,u,w],[v,w,u]]:
    #pside_deriv := [seq( subs( 
    # {side_vars[i][2]=0,side_vars[i][3]=1}, 
    # diff( subs(side_vars[i][1] = 0, polys_C1[i+6]),side_vars[i][2]) -
    # diff( subs(side_vars[i][1] = 0, polys_C1[i+6]),side_vars[i][3])), i=1..6)];
    
    
    side_vars = [
        [w, v, u],  # edge w = 0, corner u = 1  G01
        [w, u, v],  # edge w = 0, corner v = 1  G10
        [u, w, v],  # edge u = 0, corner v = 1  G12
        [u, v, w],  # edge u = 0, corner w = 1  G21
        [v, u, w],  # edge v = 0, corner w = 1  G20
        [v, w, u]]  # edge v = 0, corner u = 1  G02
    side_poly_ind = [0, 0, 1, 1, 2, 2]
    
    pside_deriv = [
        (diff(
                polys_C1[side_poly_ind[i]].subs(side_vars[i][0], 0), side_vars[i][1])
            - diff(
                polys_C1[side_poly_ind[i]].subs(side_vars[i][0], 0), side_vars[i][2])).subs(
            {side_vars[i][1]: 0, side_vars[i][2]: 1})
        for i in range(6)]
    
    
    # Midpoint derivatives in the direction of the opposite vertex
    #  e.g.,  diff( poly[0], w)- 1/2( diff(poly[1],u) - diff(poly[2],v)) evaluated at (1/2, 1/2, 0)
    # reusing side_vars here, for edges 0,1,2 - does not matter which of two vertices to pick
    
    mid_deriv = [
                                 (diff(polys_C1[i], side_vars[2 * i][0])
            - (1 / Rational(2)) * diff(polys_C1[i], side_vars[2 * i][1])
            - (1 / Rational(2)) * diff(polys_C1[i], side_vars[2 * i][2])).subs(
            {side_vars[2 * i][0]: 0, side_vars[2 * i][1]: Rational(1, 2), side_vars[2 * i][2]: Rational(1, 2)},
            )
        for i in range(3)
    ]
    polys_C1_f = [p.subs(solve( 
        [side_deriv_unknowns[i]- pside_deriv[i] for i in range(6)]+ # equations for derivatives along sides
        [mid_deriv_unknowns[i] - mid_deriv[i]   for i in range(3)] +  # equations for midpoint derivatives
        [vp[i].subs(depvars)-vtx_unknowns[i] for i in range(3)]  # corner positions
      , indepvars)) for p in polys_C1]
    return polys_C1_f


# ********* For testing: compute all element boundary data from a function f  *********

def build_data(f):
    """ Compute vertex values, edge derivatives and midpoint derivatives from a function f(u,v)
    """
    u, v = sp.symbols('u v')

    ff = sp.Lambda((u, v), f)
    df = sp.Lambda((u, v), Matrix([[sp.diff(f, u), sp.diff(f, v)]]))
    
    vtx =    sp.sympify('[[1, 0], [0, 1], [0, 0]]')
    midpt =  sp.sympify('[[1/2, 1/2], [0, 1/2], [1/2, 0]]')
    edges =  sp.sympify('[[-1, 1], [1, -1], [0, -1], [0, 1], [1, 0], [-1, 0]]')
    middir = sp.sympify('[[-1/2, -1/2], [1, -1/2], [-1/2, 1]]')
    p_values = [ff(*vtx[i]) for i in range(3)]
    grad = [df(*v) for v in vtx] 
    # gradients per endpoint of edges
    grad_e = [grad[0],grad[1],grad[1],grad[2],grad[2],grad[0]]
    G_values = [grad_e[i].dot(Matrix(edges[i])) for i in range(6)]
    N_values = [Matrix(df(*midpt[i])).dot(Matrix( middir[i])) for i in range(3)]
    return p_values + G_values + N_values

def subs_data(p, bdata):
    """ Substitute boundary data values into a polynomial expression"""
    global vtx_unknowns, side_deriv_unknowns, mid_deriv_unknowns
    subs_eqs = \
            [[vtx_unknowns[i],bdata[i]] for i in range(3)] +\
            [[side_deriv_unknowns[i], bdata[i+3]] for i in range(6)] +\
            [[mid_deriv_unknowns[i], bdata[i+9]] for i in range(3)]
    return p.subs(subs_eqs)

# ********* Bezier form *********

# currently not used, all polynomials stay in complete triangle barycentric coords, not in subtriangle
# order 003, 012,021,030, 102,111,120, 201, 210, 300
bezier_deg = 3
unity_poly = ((u+v+w)**bezier_deg).expand()
bernstein_basis = [unity_poly.coeff(u**i *v**j * w**(bezier_deg-i-j))*u**i *v**j * w**(bezier_deg-i-j)
                   for i in range(bezier_deg+1) for j in range(bezier_deg+1-i)]
monomial_basis = [  u**i *v**j * w**(bezier_deg-i-j) for i in range(bezier_deg+1) for j in range(bezier_deg+1-i)]
bernstein_scales = [ [1/unity_poly.coeff(u**i *v**j * w**(bezier_deg-i-j)) for j in range(bezier_deg+1-i)] for i in range(bezier_deg+1)]
def coeff2bezier_cubic(p): 
    return [p.coeff(u**i *v**j * w**(bezier_deg-i-j))*bernstein_scales[i][j] 
            for i in range(bezier_deg+1) for j in range(bezier_deg+1-i)]


# ******** Evaluation *********

def poly2matrix(p): 
    return Matrix( [[p.expand().coeff(u**i *v**j * w**(bezier_deg-i-j)*all_unknowns[k])
                   for i in range(bezier_deg+1) for j in range(bezier_deg+1-i)] for k in range(12)]).T

def compute_CT_matrices(polys):
    return [poly2matrix(p) for p in polys]

def triangle_ind(CTtri_bounds,u,v,w):
    """Check which triangle defined in CTtri_bounds if any contains the point (u,v,w). 
    CTtri_bounds is a list of 3 x 3 matrices of the coefficients of lines bounding subtriangles"""
    for i in range(3):
        if all( [CTtri_bounds[i][j,0]*u +CTtri_bounds[i][j,1]*v +CTtri_bounds[i][j,2]*w >= -1e-7 for j in range(len(CTtri_bounds))]):
            return i
    return -1

monomial_basis_eval = sp.lambdify( (u,v,w), Matrix(monomial_basis).evalf(),'numpy')

# This function can be easily converted to C:  just need to write out the numerical matrices 
# CT_matrices, CTtri_bounds, and write a function evaluating the monomial basis vector from u,v,w 
# (monomial_basis_eval)
def CT_eval(CT_matrices,  CTtri_bounds, boundary_data, u,v,w): 
    """
    This function evaluates the Clough-Tocher interpolant at barycentric point (u,v,w)
    It takes as input 
    -- CT_matrices: array of three 10 x 12 matrices computed from the coefficients 
    of the C-T polynomials on subtriangles using compute_CT_matrices. These are constant matrices, do not depend on data.
    -- CTtri_bounds:   3  3 x 3 matrices, one per subtriangle, each row is the coefficients of the line along an edge of the subtriangle
    -- boundary_data:  vector of 12 values for the boundary data in the order p0, p1, p2, G01, ... G02, N01, N12, N20
    """
    ind = triangle_ind(CTtri_bounds, u,v,w) 
    if ind >= 0:
        bb_vector = monomial_basis_eval(u,v,w)
        return (CT_matrices[ind]*boundary_data).dot(bb_vector)
    else:
        return np.nan

# *******  Converting to Lagrange basis ****** 
def generate_L2L(polys_C1, node_bary, node_subtri, all_inknowns): 
    """
    Generates a matrix L_d2L that maps a vector of 12 dofs (pi, Gij, Nij) to the vector of Lagrange node values 
    enumerated as described  by node_bary and node_vars. node_subtri is the 3 polynomials on subtriangles with 
    coefficients given in terms of (pi, Gij, Nij)
    """
    subtri_polys  = [Poly(p,[u,v,w]) for p in polys_C1]
    Lagr_node_vals = [ subtri_polys[node_subtri[i]](*(node_bary[i]/9)) for i in range(19) ]
    local_dof_vars = all_unknowns
    L_d2L = Matrix( [ [ Lnode.coeff(v) for v in local_dof_vars]  for Lnode in Lagr_node_vals])
    return L_d2L

# a vector of coefficients c_e

def generate_ce(subtri_polys):
    "Creates a vector of coefficients which, when applied to the vector [p0,p1,G01,G10,N01] yields the derivative along the edge at the midpoint."
    midpt_dfde = subtri_polys[0].as_expr().subs({w:0,v:1-u}).expand().collect(u).diff(u).subs({u:Rational(1,2)})
    edge_endpt_vars = [p0,p1,G01,G10, N01]
    c_e = [midpt_dfde.coeff(v) for v in edge_endpt_vars]
    return c_e

# ******** Testing *********

def runTests(polys_C1_f):
    global pt, center;
    q300, q030, q210, q120, q201, q021, q003, q111, q102, q012 = \
            symbols('q300 q030 q210 q120 q201 q021 q003 q111 q102 q012')
    testpoly = q300 * u**3 + q030 * v**3 + q210 * u**2 * v + q120 * u * v**2 \
                    + q201 * u**2 * w + q021 * v**2 * w + q003 * w**3 + q111 * u * v * w \
                    + q102 * u * w**2 + q012 * v * w**2
    
    # Test: infer the input dofs from a polynomial (using generic deg 3) and verify that the result is the same, should be [0,0,0,0]
    bd = build_data(testpoly.subs(w,1-u-v))
    passed = not any([subs_data(p-testpoly,bd) for p in polys_C1_f ])
    print('Cubic polynomial reproduction '+ ('passed' if passed else 'not passed'))    
    
    # Test that polynomials on boundaries between subtriangles match 
    passed = not any ([ (polys_C1_f[j].subs( dict( [[uvwvars[i], ep2param_line(pt[j],center)[i]] for i in range(3)]))-\
    polys_C1_f[prev(j)].subs( dict( [[uvwvars[i], ep2param_line(pt[j],center)[i]] for i in range(3)]))).expand() for j in range(3)])
    print('Polynomial on internal subtriangle boundaries match '+ ('passed' if passed else 'not passed'))    
    
    # Test that derivatives of polynomials on boundaries between subtriangles match 
    hom_grads = [gradb(polys_C1_f[i]) for i in range(3)]
    passed = not any([ (hom_grads[j][k].subs( dict( [[uvwvars[i], ep2param_line(pt[j],center)[i]] for i in range(3)]))-\
      hom_grads[prev(j)][k].subs( dict( [[uvwvars[i], ep2param_line(pt[j],center)[i]] for i in range(3)]))).expand() 
           for j in range(3) for k in range(3)])
    print('Gradient on internal subtriangle boundaries match '+ ('passed' if passed else 'not passed'))    



def test_Lagrange_consistency(polys_C1_f, L_L2d_ind, L_ind2dep): 
    """
    polys_C1_f: 3 subtriangle polynomials with coefficients in terms of (pi, Gij, Nij) variables, 
    L_L2d_ind:  12 x12  matrix expressing (pi, Gij, Nij) in terms of 12 indep. Lagrange node values 
    L_ind2dep:  matrix expressing 7 dependence Lagrange nodes in terms of independent; this ensures internal edge C1 constraints 
    The test verifies that 
    (1) when we use   L_L2d_ind to express polynomial coefficient in terms of indep Lagrange nodes 
    we get these nodes back when evaluating at their barycentric coordinates. 
    (2)  when we evaluate at the dependent node coordinates we get the same as we get by applying L_ind2dep to indep  node values. 
    As for any choice of (pi, Gij, Nij) dofs the polynomials are already verified to be C1 at the interior edges, 
    this also verifies that if the dependent node values are computed from independent using L_ind2dep, the resulting polynomials
    are C1 at the interior edges. 
    """
    global all_unknowns, node_vars, node_bary, subtri_nodes
    dof_vars = all_unknowns

    # express the triangle dof values (pi, Gij, Nij) in terms of independ lagrange node values using provided matrix
    dof_lagr_subs = dict( [[ dof_vars[i], (L_L2d_ind *  Matrix( [node_vars[0:12]]).T)[i] ] for i in range(12)])
    # express dependent node values in terms of indepedent
    dep_nodes_expr = L_ind2dep*Matrix([node_vars[0:12]]).T

    # evaluate the polynomials on subtriangles at node barycentric coordinates, verify we get the Lagrange nodes back
    passed = True
    for i in range(3): 
        passed = passed and not any([ Poly( polys_C1_f[i].subs( dof_lagr_subs), [u,v,w])(*node_bary[j]/9)- node_vars[j] for j in subtri_nodes[i] if j < 12]+\
        [ Poly(  polys_C1_f[i].subs( dof_lagr_subs), [u,v,w])(*node_bary[j]/9)- dep_nodes_expr[j-12] for j in subtri_nodes[i] if j >= 12])
    print('Lagrange consistency test '+ ('passed' if passed else 'not passed'))    

#  ********  Plotting  *********
# !!!! ChatGPT generated - probably there is a better way to plot a function on a triangle but not sure 
import plotly.graph_objects as go

def plot_function_on_triangle(func, fun_name, vertices, resolution=20):
    """
    Plots a function on a single triangular domain using a regular grid.

    Args:
        func (callable): A function of two arguments (u, v) to be plotted.
        vertices (list): A list of 3 corner points defining the triangle in the (u, v)-plane, e.g., [[0, 0], [1, 0], [0, 1]].
        resolution (int): Number of divisions along each edge of the triangle.

    Returns:
        None. Displays an interactive plot.
    """
    # Convert vertices to a numpy array
    vertices = np.array(vertices)

    # Create a regular grid of barycentric coordinates
    bary_coords, triangles = create_regular_grid_and_triangles(resolution)

    # Convert barycentric coordinates to Cartesian coordinates
    grid_points = barycentric_to_cartesian(bary_coords, vertices)
    
    # Evaluate the function at grid points
    z_vals = np.array([float(func(u, v)) for u, v in grid_points])

    # Add the function surface as a triangulated mesh
    fig = go.Figure(go.Mesh3d(
        x=grid_points[:, 0],
        y=grid_points[:, 1],
        z=z_vals,
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        intensity=z_vals,
        colorscale="Viridis",
        colorbar=dict(title="Function Value"),
        opacity=0.9
    ))

    # Add the triangle base in the (u, v)-plane for context
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=[0, 0, 0],
        i=[0],
        j=[1],
        k=[2],
        color="lightblue",
        opacity=0.5,
        name="Triangle Base"
    ))

    # Update the layout for better visualization
    fig.update_layout(
        title="Clough-Tocher interpolant of "+fun_name,
        scene=dict(
            xaxis_title="u",
            yaxis_title="v",
            zaxis_title="Function Value",
            aspectratio=dict(x=1, y=1, z=0.5),
        ),
        width=1000,
        height=1000
    )
    # Show the interactive plot
    fig.show()
    return fig


def create_regular_grid_and_triangles(resolution):
    """
    Creates a regular grid of points and corresponding triangles inside a triangular domain.

    Args:
        resolution: Number of divisions along each edge.

    Returns:
        tuple:
            - np.ndarray: Array of barycentric coordinates for grid points.
            - np.ndarray: Array of triangles, where each row contains 3 vertex indices.
    """
    bary_coords = []
    triangles = []

    # Generate barycentric coordinates for grid points
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            bary_coords.append([i / resolution, j / resolution, k / resolution])
            s = bary_coords[-1][0]+ bary_coords[-1][1]+ bary_coords[-1][2]
            if s > 1: 
                print(s)

    # Generate triangle indices
    for i in range(resolution):
        for j in range(resolution - i):
            # Index of the top-left vertex
            idx = i * (resolution + 1) - (i * (i - 1)) // 2 + j
            # Triangle 1
            triangles.append([idx, idx + 1, idx + resolution + 1 - i])
            # Triangle 2
            if j < resolution - i - 1:
                triangles.append([idx + 1, idx + resolution + 2 - i, idx + resolution + 1 - i])

    return np.array(bary_coords), np.array(triangles)


def barycentric_to_cartesian(bary_coords, vertices):
    """
    Converts barycentric coordinates to Cartesian coordinates.

    Args:
        bary_coords: Barycentric coordinates as a numpy array.
        vertices: Triangle vertices as a numpy array.

    Returns:
        np.ndarray: Cartesian coordinates of the points.
    """
    return np.dot(bary_coords, vertices)

# end ChatGpt code

if __name__ == '__main__':
    # run validation
    polys_C1_f = derive_C1_polynomials(vtx_unknowns,side_deriv_unknowns,mid_deriv_unknowns)
    runTests(polys_C1_f)
