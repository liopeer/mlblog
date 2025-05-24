using Plots, Flux, LinearAlgebra, CUDA, ProgressMeter, MLUtils
using Flux: @functor
using Statistics: mean
using MLDatasets: MNIST

struct ResNetBlock
    conv1::Conv
    conv2::Conv
    skipconv::Conv
end

function ResNetBlock(in_channels::Integer, out_channels::Integer)
    ResNetBlock(
        Conv((3,3), in_channels => out_channels, pad=SamePad()),
        Conv((3,3), out_channels => out_channels, pad=SamePad()),
        Conv((1,1), in_channels => out_channels, pad=SamePad())
    )
end

function (m::ResNetBlock)(x::AbstractArray{Float32, 4})
    skip = m.skipconv(x)
    x = relu(m.conv1(x))
    x = m.conv2(x) + skip
    return relu(x)
end

@functor ResNetBlock

struct ResNetMNIST
    convs::Vector{ResNetBlock}
    pool::MaxPool
    fc::Dense
end

function ResNetMNIST()
    conv1 = ResNetBlock(1, 64)
    conv2 = ResNetBlock(64, 128)
    conv3 = ResNetBlock(128, 256)
    pool = MaxPool((2,2), pad=SamePad())
    out_feat = 256 * 4 * 4
    fc = Dense(out_feat => 10)
    return ResNetMNIST(
        [conv1, conv2, conv3],
        pool,
        fc
    )
end

function (m::ResNetMNIST)(x::AbstractArray{Float32, 4})
    for conv in m.convs
        x = m.pool(conv(x))
    end
    bs = size(x, ndims(x))
    other = div(length(x), size(x,4))
    return m.fc(reshape(x, (other, bs)))
end

@functor ResNetMNIST

bs = 8
train_ds = MNIST(:train)
test_ds = MNIST(:test)

train_loader = MLUtils.DataLoader(train_ds, batchsize=bs, shuffle=true)
test_loader = MLUtils.DataLoader(test_ds, batchsize=bs)

model = ResNetMNIST() |> gpu 
optim = Flux.setup(Flux.Adam(0.0001), model)

epochs = 30
train_losses = []
test_losses = []
classes = 0:9

@showprogress for epoch in 1:epochs
    train_losses_step = []
    for (i,x) in enumerate(train_loader)
        img = x.features |> gpu
        img = reshape(img, (28, 28, 1, size(img)[end]))
        target = x.targets
        target = Flux.onehotbatch(target, classes) |> gpu

        loss(m) = Flux.logitcrossentropy(m(img), target)

        loss, grads = Flux.withgradient(loss, model)
        
        Flux.update!(optim, model, grads[1])
        push!(train_losses_step, loss |> cpu)
    end
    push!(train_losses, mean(train_losses_step))

    test_losses_step = []
    for (i,x) in enumerate(test_loader)
        img = x.features |> gpu
        img = reshape(img, (28, 28, 1, size(img)[end]))
        target = x.targets
        target = Flux.onehotbatch(target, classes) |> gpu

        loss(m) = Flux.logitcrossentropy(m(img), target)

        loss = loss(model)
        
        push!(test_losses_step, loss |> cpu)
    end
    push!(test_losses, mean(test_losses_step))
end