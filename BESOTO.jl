using LinearAlgebra
using SparseArrays
using Printf
using WriteVTK

# ============================================================================
# 1. FUNCIONES DE EXPORTACIÓN (ParaView)
# ============================================================================
function exportar_paso_vtk(x, iter, nx, ny, nz, carpeta="resultados_beso")
    mkpath(carpeta)
    diseno_3d = reshape(x, (nx, ny, nz))
    nombre_archivo = joinpath(carpeta, @sprintf("frame_%03d", iter))
    
    x_c = collect(0.0:Float64(nx))
    y_c = collect(0.0:Float64(ny))
    z_c = collect(0.0:Float64(nz))
    
    vtk_grid(nombre_archivo, x_c, y_c, z_c) do vtk
        vtk["Densidad"] = diseno_3d
    end
end

# ============================================================================
# 2. MATRIZ DE RIGIDEZ ELEMENTAL (HEX8)
# ============================================================================
function hex8_stiffness(E, ν)
    g = 1.0 / sqrt(3.0)
    gp = [-g, g]
    C = E / ((1.0+ν)*(1.0-2.0*ν)) * [
        1.0-ν   ν     ν     0.0         0.0         0.0;
        ν     1.0-ν   ν     0.0         0.0         0.0;
        ν     ν     1.0-ν   0.0         0.0         0.0;
        0.0     0.0     0.0     (1.0-2.0*ν)/2.0   0.0         0.0;
        0.0     0.0     0.0     0.0         (1.0-2.0*ν)/2.0   0.0;
        0.0     0.0     0.0     0.0         0.0         (1.0-2.0*ν)/2.0
    ]

    Ke = zeros(Float64, 24, 24)
    for gx in gp, gy in gp, gz in gp
        dN = 1.0/8.0 * [
            -(1-gy)*(1-gz)  (1-gy)*(1-gz)  (1+gy)*(1-gz)  -(1+gy)*(1-gz)  -(1-gy)*(1+gz)  (1-gy)*(1+gz)  (1+gy)*(1+gz)  -(1+gy)*(1+gz);
            -(1-gx)*(1-gz)  -(1+gx)*(1-gz)  (1+gx)*(1-gz)  (1-gx)*(1-gz)  -(1-gx)*(1+gz)  -(1+gx)*(1+gz)  (1+gx)*(1+gz)  (1-gx)*(1+gz);
            -(1-gx)*(1-gy)  -(1+gx)*(1-gy)  -(1+gx)*(1+gy)  -(1-gx)*(1+gy)  (1-gx)*(1-gy)  (1+gx)*(1-gy)  (1+gx)*(1+gy)  (1-gx)*(1+gy)
        ]
        J = dN * [0 0 0; 1 0 0; 1 1 0; 0 1 0; 0 0 1; 1 0 1; 1 1 1; 0 1 1]
        dNdx = J \ dN
        B = zeros(Float64, 6, 24)
        for i in 1:8
            B[:, 3*i-2:3*i] = [
                dNdx[1,i] 0.0 0.0; 0.0 dNdx[2,i] 0.0; 0.0 0.0 dNdx[3,i];
                dNdx[2,i] dNdx[1,i] 0.0; 0.0 dNdx[3,i] dNdx[2,i]; dNdx[3,i] 0.0 dNdx[1,i]
            ]
        end
        Ke += B' * C * B * det(J)
    end
    return Ke
end

# ============================================================================
# 3. CONSTRUCCIÓN DE MALLA Y FILTRO
# ============================================================================
function precomputar_malla_y_filtro(nx, ny, nz, rmin)
    nelems = nx * ny * nz
    edof = zeros(Int, nelems, 24)
    nnx, nny = nx + 1, ny + 1

    for k in 1:nz, j in 1:ny, i in 1:nx
        eid = i + (j-1)*nx + (k-1)*nx*ny
        n1 = i + (j-1)*nnx + (k-1)*nnx*nny
        nodes = [n1, n1+1, n1+1+nnx, n1+nnx, n1+nnx*nny, n1+1+nnx*nny, n1+1+nnx+nnx*nny, n1+nnx+nnx*nny]
        for (ln, gn) in enumerate(nodes)
            edof[eid, 3*ln-2:3*ln] = [3*gn-2, 3*gn-1, 3*gn]
        end
    end

    H_I = Int[]; H_J = Int[]; H_V = Float64[]
    r = ceil(Int, rmin)
    for k1=1:nz, j1=1:ny, i1=1:nx
        e1 = i1 + (j1-1)*nx + (k1-1)*nx*ny
        for k2=max(1, k1-r):min(nz, k1+r), j2=max(1, j1-r):min(ny, j1+r), i2=max(1, i1-r):min(nx, i1+r)
            e2 = i2 + (j2-1)*nx + (k2-1)*nx*ny
            dist = sqrt((i1-i2)^2 + (j1-j2)^2 + (k1-k2)^2)
            if dist <= rmin
                push!(H_I, e1); push!(H_J, e2); push!(H_V, rmin - dist)
            end
        end
    end
    H = sparse(H_I, H_J, H_V)
    Hs = sum(H, dims=2)[:]

    I_sparse = zeros(Int, 24*24*nelems)
    J_sparse = zeros(Int, 24*24*nelems)
    idx = 1
    for e in 1:nelems
        for i in 1:24, j in 1:24
            I_sparse[idx] = edof[e,i]
            J_sparse[idx] = edof[e,j]
            idx += 1
        end
    end

    return edof, H, Hs, I_sparse, J_sparse
end

# ============================================================================
# 4. SOFTWARE CORE BESO (CPU)
# ============================================================================
# CAMBIO 3: rmin sube a 3.0 por defecto
function run_beso_3d(; nx=40, ny=20, nz=10, volfrac=0.4, ER=0.02, rmin=3.0, maxiter=100)
    println("--- INICIANDO SOFTWARE BESO 3D ROBUSTO ---")
    println("Malla: $nx x $ny x $nz | Vol. Objetivo: $(volfrac*100)% | ER: $(ER*100)% | Rmin: $rmin")

    nelems = nx * ny * nz
    ndofs = 3 * (nx+1) * (ny+1) * (nz+1)
    
    x = ones(Float64, nelems) 
    # CAMBIO 5: Vacío más estable
    x_min = 1e-4 
    vol_current = 1.0

    Ke0 = hex8_stiffness(1.0, 0.3)
    edof, H, Hs, I_sparse, J_sparse = precomputar_malla_y_filtro(nx, ny, nz, rmin)

    # --- CAMBIO 1: CREAR DOMINIOS NO-DISEÑO (Elementos protegidos) ---
    passive_solid = Int[]
    for k in 1:nz, j in 1:ny, i in 1:nx
        e = i + (j-1)*nx + (k-1)*nx*ny
        # Proteger las 2 primeras capas de la izquierda (Soportes)
        if i <= 2
            push!(passive_solid, e)
        end
        # Proteger área circundante a la carga (derecha, centro Y, centro Z)
        if i >= nx-1 && abs(j - div(ny,2)) <= 2 && abs(k - div(nz,2)) <= 2
            push!(passive_solid, e)
        end
    end

    # Condiciones de Contorno (Voladizo: Cara izquierda fija)
    fixed_dofs = Int[]
    for j in 1:(ny+1), k in 1:(nz+1)
        node = 1 + (j-1)*(nx+1) + (k-1)*(nx+1)*(ny+1)
        push!(fixed_dofs, 3*node-2, 3*node-1, 3*node)
    end
    free_dofs = setdiff(1:ndofs, fixed_dofs)

    # --- CAMBIO 2: CARGAS DISTRIBUIDAS ---
    F = zeros(Float64, ndofs)
    # Distribuir la carga en un parche de nodos en lugar de uno solo
    for j in div(ny,2)-1:div(ny,2)+2, k in div(nz,2)-1:div(nz,2)+2
        load_node = nx+1 + (j-1)*(nx+1) + (k-1)*(nx+1)*(ny+1)
        F[3*load_node-1] = -1.0 # Fuerza en Y negativo
    end
    F_free = F[free_dofs]

    u = zeros(Float64, ndofs)
    sens_history = zeros(Float64, nelems)
    c_history = Float64[]

    for iter in 1:maxiter
        
        V_sparse = zeros(Float64, 24*24*nelems)
        idx = 1
        for e in 1:nelems
            E_val = x[e] == 1.0 ? 1.0 : x_min
            for i in 1:24, j in 1:24
                V_sparse[idx] = E_val * Ke0[i, j]
                idx += 1
            end
        end
        K = sparse(I_sparse, J_sparse, V_sparse, ndofs, ndofs)
        
        u[free_dofs] = K[free_dofs, free_dofs] \ F_free

        sens = zeros(Float64, nelems)
        ce_total = 0.0
        for e in 1:nelems
            ue = u[edof[e, :]]
            ce = 0.5 * dot(ue, Ke0 * ue) 
            sens[e] = ce
            ce_total += x[e] == 1.0 ? ce : (ce * x_min)
        end
        push!(c_history, ce_total)

        sens_filtered = (H * sens) ./ Hs

        # --- CAMBIO 4: ESTABILIZACIÓN HISTÓRICA MÁS FUERTE ---
        if iter > 1
            sens_filtered = 0.4 .* sens_filtered .+ 0.6 .* sens_history
        end
        sens_history .= sens_filtered

        # --- CAMBIO 1 (Continuación): PROTEGER DOMINIOS NO-DISEÑO ---
        # Fuerza a los elementos críticos a estar primeros en la lista de ordenamiento
        sens_filtered[passive_solid] .= Inf

        vol_target = max(vol_current * (1.0 - ER), volfrac)
        target_solid_elems = round(Int, vol_target * nelems)

        # Actualización Discreta
        sorted_indices = sortperm(sens_filtered, rev=true)
        fill!(x, x_min) 
        solid_indices = sorted_indices[1:target_solid_elems]
        x[solid_indices] .= 1.0 
        
        vol_current = sum(x .== 1.0) / nelems

        exportar_paso_vtk(x, iter, nx, ny, nz, "evolucion_beso_3d")

        @printf("Iter: %3d | Vol: %.3f | Compliance: %.4e\n", iter, vol_current, ce_total)

        if vol_current <= volfrac && iter > 10
            error_c = abs(c_history[end] - c_history[end-1]) / c_history[end]
            if error_c < 0.001
                println("✅ Convergencia alcanzada en la iteración $iter.")
                break
            end
        end
    end

    println("Archivos VTK exportados en la carpeta: /evolucion_beso_3d/")
    return x, c_history
end

# ============================================================================
# EJECUCIÓN
# ============================================================================
# Rmin=3.0 garantiza ramas gruesas, Malla ligeramente mayor para ver detalle
x_final, history = run_beso_3d(nx=60, ny=20, nz=10, volfrac=0.3, ER=0.02, rmin=3.0, maxiter=100)
