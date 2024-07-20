using Flux, ProgressMeter, Plots, Metal, LinearAlgebra
using Flux: @functor
# using MLDatasets: MNIST
include("unet.jl")

"""Timestep Embedding."""
struct TimeEmbed
    emb::Array{Float32, 2}
end

function TimeEmbed(timesteps::Integer, embed_dim::Integer)
    emb = zeros(embed_dim, timesteps)
    for t in 1:timesteps
        for dim in 1:div(embed_dim, 2)
            omega = 1 / (10000.0^(2*dim/embed_dim))
            emb[2*dim, t] = sin(t * omega)
            emb[2*dim-1, t] = cos(t * omega)
        end
    end
    TimeEmbed(emb)
end

function (emb::TimeEmbed)(t::Vector{<:Integer})
    return emb.emb[:,t]
end


"""Diffusion Process."""
struct Diffusion
    timesteps::Integer
    betas::Vector{Float32}
    alphas::Vector{Float32}
    one_minus_alphas::Vector{Float32}
    sqrt_alphas::Vector{Float32}
    alphas_bar::Vector{Float32}
    one_minus_alphas_bar::Vector{Float32}
    sqrt_one_minus_alphas_bar::Vector{Float32}
    unet::UNet
    t_emb::TimeEmbed
end

function Diffusion(
    timesteps::Integer,
    unet::UNet,
    t_emb::TimeEmbed,
    betastart::AbstractFloat = 0.0001,
    betaend::AbstractFloat = 0.02,
)
    betas = Float32.(LinRange(betastart, betaend, timesteps))
    alphas = 1.0 .- betas
    alphas_bar = cumprod(alphas)

    return Diffusion(
        timesteps, 
        betas, 
        alphas, 
        1 .- alphas,
        sqrt.(alphas),
        alphas_bar, 
        1 .- alphas_bar,
        sqrt.(1 .- alphas_bar),
        unet,
        t_emb
    )
end
@functor Diffusion

function add_img_dims(vec::Vector{<:Any})
    return reshape(vec, (1, 1, 1, size(vec)...))
end

function denoise_ddpm(
    diffusion::Diffusion, 
    x::Array{Float32, 4}, 
    t::Array{Int64, 1}
)
    sqrt_alphas = add_img_dims(diffusion.sqrt_alphas[t])
    one_minus_alphas = add_img_dims(diffusion.one_minus_alphas[t])
    sqrt_one_minus_alphas_bar = add_img_dims(diffusion.sqrt_one_minus_alphas_bar[t])
    sigmas = add_img_dims(sqrt.(diffusion.betas[t]))

    t = diffusion.t_emb(t)
    eps = diffusion.unet(x, t)
    noise = randn(size(eps)...)
    @. x = 1 / sqrt_alphas * (x - one_minus_alphas / sqrt_one_minus_alphas_bar * eps) + sigmas * noise
end

if abspath(PROGRAM_FILE) == @__FILE__
    bs, c, h, w, t_dim = 16, 3, 64, 64, 256

    t_steps = 1000

    t_emb = TimeEmbed(t_steps, t_dim)
    unet = UNet(c, c, 64, [1, 2, 4, 8], t_dim)

    diff = Diffusion(t_steps, unet, t_emb) |> gpu

    x = Float32.(randn(w, h, c, bs)) |> gpu
    t = rand(1:t_steps, bs) |> gpu
    denoise_ddpm(diff, x, t)
end