using Flux: @functor, Conv, relu, ConvTranspose, SamePad, Dense

"""Residual Block with Timestep Conditioning."""
struct ResNetBlock
    conv1::Conv
    conv2::Conv
    skipconv::Conv
    conditioning::Dense
end

function ResNetBlock(in_channels::Integer, out_channels::Integer, t_embed_dim::Integer)
    ResNetBlock(
        Conv((3,3), in_channels => out_channels, pad=SamePad()),
        Conv((3,3), out_channels => out_channels, pad=SamePad()),
        Conv((1,1), in_channels => out_channels),
        Dense(t_embed_dim => out_channels)
    )
end

function (m::ResNetBlock)(x::Array{Float32, 4}, t::Array{Float32, 2})
    t = m.conditioning(t)
    t = reshape(t, (1, 1, size(t)...))

    skip = m.skipconv(x)

    x = relu(m.conv1(x))
    x = x .+ t
    x = m.conv2(x) + skip
    return relu(x)
end
@functor ResNetBlock


"""Downsampling Layer."""
struct DownSample
    conv::Conv
end

function DownSample(channels::Integer)
    DownSample(
        Conv((3,3), channels => channels, pad=SamePad(), stride=2)
    )
end

function (m::DownSample)(x::Array{Float32, 4})
    return relu(m.conv(x))
end
@functor DownSample


"""Upsampling Layer."""
struct UpSample
    conv::ConvTranspose
end

function UpSample(channels::Integer)
    UpSample(
        ConvTranspose((3,3), channels => channels, pad=SamePad(), stride=2)
    )
end

function (m::UpSample)(x::Array{Float32, 4})
    return relu(m.conv(x))
end
@functor UpSample


"""Residual Block with Downsampling and Timestep Conditioning."""
struct DownSampleBlock
    resblock::ResNetBlock
    downsample::DownSample
end

function DownSampleBlock(in_channels::Integer, out_channels::Integer, t_embed_dim::Integer)
    DownSampleBlock(
        ResNetBlock(in_channels, out_channels, t_embed_dim),
        DownSample(out_channels)
    )
end

function (m::DownSampleBlock)(x::Array{Float32, 4}, t::Array{Float32, 2})
    return m.downsample(m.resblock(x, t))
end
@functor DownSampleBlock


"""Residual Block with Upsampling and Timestep Conditioning"""
struct UpSampleBlock
    upsample::UpSample
    resblock::ResNetBlock
end

function UpSampleBlock(in_channels::Integer, out_channels::Integer, t_embed_dim::Integer)
    UpSampleBlock(
        UpSample(in_channels),
        ResNetBlock(in_channels, out_channels, t_embed_dim)
    )
end

function (m::UpSampleBlock)(x::Array{Float32, 4}, t::Array{Float32, 2})
    return m.resblock(m.upsample(x), t)
end
@functor UpSampleBlock


"""UNet with Timestep Conditioning."""
struct UNet
    in_conv::Conv
    downsample_blocks::Vector{DownSampleBlock}
    upsample_blocks::Vector{UpSampleBlock}
    out_conv::Conv
end

function UNet(
    in_channels::Integer,
    out_channels::Integer,
    base_channels::Integer,
    channel_multipliers::Vector{<:Integer},
    t_embed_dim::Integer
)
    in_conv = Conv((3,3), in_channels => base_channels, pad=SamePad())

    channels = base_channels .* channel_multipliers
    @assert channels[1] == base_channels

    downblocks = []
    for (in_ch, out_ch) in zip(channels[1:end-1], channels[2:end])
        push!(downblocks, DownSampleBlock(in_ch, out_ch, t_embed_dim))
    end

    channels = 2 .* reverse(channels)

    upblocks = []
    for (in_ch, out_ch) in zip(channels[begin:end-1], channels[begin+1:end])
        push!(upblocks, UpSampleBlock(in_ch, div(out_ch, 2), t_embed_dim))
    end

    out_conv = Conv((3,3), div(channels[end], 2) => out_channels, pad=SamePad())

    return UNet(in_conv, downblocks, upblocks, out_conv)
end

function (m::UNet)(x::Array{Float32, 4}, t::Array{Float32, 2})
    skips = []

    x = m.in_conv(x)
    push!(skips, x)

    for block in m.downsample_blocks
        x = block(x, t)
        push!(skips, x)
    end

    for (i, block) in enumerate(m.upsample_blocks)
        x = cat(x, reverse(skips)[i], dims=3)
        x = block(x, t)
    end
    m.out_conv(x)
end
@functor UNet