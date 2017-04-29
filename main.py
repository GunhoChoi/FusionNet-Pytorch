from FusionNet import * 
import numpy as np
import matplotlib.pyplot as plt

# hyperparameters

batch_size = 16
img_size = 512
lr = 0.0002
epoch = 1000

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
    for _,(image,label) in enumerate(img_batch):
        satel_image, map_image = torch.chunk(image, chunks=2, dim=3) 
        
        # quarter size image

        satel_left,satel_right = torch.chunk(satel_image,chunks=2, dim=3)
        satel_0, satel_1 = torch.chunk(satel_left,chunks=2,dim=2)
        satel_2, satel_3 = torch.chunk(satel_right,chunks=2,dim=2)

        map_left,map_right = torch.chunk(map_image,chunks=2, dim=3)
        map_0, map_1 = torch.chunk(map_left,chunks=2,dim=2)
        map_2, map_3 = torch.chunk(map_right,chunks=2,dim=2)

        h,w = satel_0.size()[2:]

        satel_list = [satel_0, satel_1, satel_2, satel_3]
        map_list = [map_0, map_1,map_2, map_3]

        for idx in range(4):    

            optimizer.zero_grad()

            x = Variable(satel_list[idx]).cuda()
            y_ = Variable(map_list[idx]).cuda()
            y = fusion.forward(x)
            
            loss = loss_func(y,y_)
            loss.backward()
            optimizer.step()

        if _ % 500 ==0:
            print(i)
            print(loss)
            #print(y.size())
            v_utils.save_image(x[0].cpu().data,"./result/satel_image_{}_{}.png".format(i,_))
            v_utils.save_image(y_[0].cpu().data,"./result/map_image_{}_{}.png".format(i,_))
            v_utils.save_image(y[0].cpu().data,"./result/gen_image_{}_{}.png".format(i,_))
            torch.save(fusion,"./model/fusion.pkl")    