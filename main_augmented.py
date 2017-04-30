from FusionNet import * 
import numpy as np
import matplotlib.pyplot as plt

# hyperparameters

batch_size = 16
img_size = 512
slice_size = 256
lr = 0.0002
epoch = 200

# input pipeline

img_dir = "./maps/"
img_data = dset.ImageFolder(root=img_dir, transform = transforms.Compose([
                                            transforms.Scale(size=img_size),
                                            transforms.CenterCrop(size=(img_size,img_size*2)),
                                            transforms.ToTensor(),
                                            ]))
img_batch = data.DataLoader(img_data, batch_size=batch_size,
                            shuffle=True, num_workers=2)

# initiate FusionNet

fusion = nn.DataParallel(FusionNet()).cuda()

try:
    fusion = torch.load('./model/fusion.pkl')
    print("\n--------model restored--------\n")
except:
    print("\n--------model not restored--------\n")
    pass

# loss function & optimizer

loss_func = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(fusion.parameters(),lr=lr)

# training

for i in range(epoch):
    for j,(image,label) in enumerate(img_batch):
        satel_image, map_image = torch.chunk(image, chunks=2, dim=3) 
        
        for h in range(slice_size):
            for w in range(slice_size):

                satel_img = satel_image[:,:,h:h+slice_size,w:w+slice_size]
                map_img = map_image[:,:,h:h+slice_size,w:w+slice_size]

                optimizer.zero_grad()

                x = Variable(satel_img).cuda()
                y_ = Variable(map_img).cuda()
                y = fusion.forward(x)
                
                loss = loss_func(y,y_)
                loss.backward()
                optimizer.step()

            print(loss)

            v_utils.save_image(x.cpu().data,"./result/satel_image_{}_{}.png".format(i,j))
            v_utils.save_image(y_.cpu().data,"./result/map_image_{}_{}.png".format(i,j))
            v_utils.save_image(y.cpu().data,"./result/gen_image_{}_{}.png".format(i,j))

            
            torch.save(fusion,"./model/fusion.pkl")    
