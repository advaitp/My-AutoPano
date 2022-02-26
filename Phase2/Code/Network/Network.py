"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import tensorflow as tf
import sys
import numpy as np
import tensorflow.keras as keras
# Don't generate pyc codes
sys.dont_write_bytecode = True

class myModel(keras.Model):
    def __init__(self):
      super(myModel,self).__init__()
      self.conv1=keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu',padding='same')
      self.conv2=keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu',padding='same')
      self.conv2_1=keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu',padding='same')
      self.BN1=keras.layers.BatchNormalization(trainable=True)
      self.BN1_1=keras.layers.BatchNormalization(trainable=True)
      self.BN1_2=keras.layers.BatchNormalization(trainable=True)
      self.BN1_3=keras.layers.BatchNormalization(trainable=True)
      self.conv2_2=keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu',padding='same')
      self.conv2_3=keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu',padding='same')
      self.maxpool=keras.layers.MaxPooling2D(pool_size=2,strides=2)
      self.maxpool2=keras.layers.MaxPooling2D(pool_size=2,strides=2)
      self.maxpool3=keras.layers.MaxPooling2D(pool_size=2,strides=2)

      self.conv3= keras.layers.Conv2D(filters=128,kernel_size=3,activation='relu',padding='same')
      self.conv4=keras.layers.Conv2D(filters=128,kernel_size=3,activation='relu',padding='same')
      self.conv4_2=keras.layers.Conv2D(filters=128,kernel_size=3,activation='relu',padding='same')
      self.conv4_3=keras.layers.Conv2D(filters=128,kernel_size=3,activation='relu',padding='same')
      self.BN2=keras.layers.BatchNormalization(trainable=True)
      self.BN2_2=keras.layers.BatchNormalization(trainable=True)
      self.BN2_3=keras.layers.BatchNormalization(trainable=True)
      self.BN2_4=keras.layers.BatchNormalization(trainable=True)
      self.global_pool=keras.layers.GlobalAveragePooling2D()
      self.loss=self.photometric_loss
      #self.flat=nn.Flatten()
      self.linear1=keras.layers.Dense(1024,activation='relu')
      self.linear2=keras.layers.Dense(8,activation=None, kernel_initializer='zeros')
      self.num_batches=32
      #returnself.__call()
      
    def photometric_loss(self,y_true,y_pred):
        return (tf.math.reduce_mean(tf.math.abs(tf.math.subtract(y_true,y_pred))))
    def epe_loss(self,y_true,y_pred):
       return (tf.math.reduce_mean(tf.math.square(tf.math.subtract(y_true,y_pred))))
    def unsupervised_model(self,x):
        input_1=x[0]
        input_2=x[1]
        
        x=self.supervised_model(input_1)
        #num_batch=x.shape[0]
        Xs=tf.stack([[42,42],[42,170],[170,170],[170,42]])
        Xs=tf.tile(tf.expand_dims(tf.reshape(Xs,-1),0),tf.stack([tf.shape(x)[0],1]))
        Xs=tf.cast(Xs,tf.float32)
        #x=tf.reshape(x,t)
        Xp=Xs+x
        H=self.homography(Xs,x)
        H=tf.linalg.pinv(H)
        batch_size=tf.shape(H)[0]
        w=tf.shape(H)[1]
        w=tf.cast(w,dtype=tf.float32)
        h=tf.shape(H)[2]
        h=tf.cast(h,dtype=tf.float32)
        M=tf.stack([[w/2.0, 0.0 ,w/2.0],[0. , h/2.0 , h/2.0],[0.,0.,1.0]])
        #M=tf.cast()
        M_t=tf.tile(tf.expand_dims(M,0),[batch_size,1,1])
        M_inv_t=tf.linalg.inv(M_t)
        inverse_homography=M_inv_t @ H @ M_t
        out_size=(input_1.shape[1],input_1.shape[2])
        output=self.batch_transform(input_2,inverse_homography,out_size,name='BatchSpatialTransformer')
        return output
    def supervised_model(self,x):
       
       x=self.conv1(x)
       
       x=self.BN1(x)
    
       x=self.conv2(x)
       
       x=self.BN1_1(x)
       
       x=self.maxpool(x)
    
       x=self.conv2_2(x)
       
       x=self.BN1_2(x)
    
       x=self.conv2_3(x)
       
       x=self.BN1_3(x)
       x=self.maxpool2(x)
    
       x=self.conv3(x)
       
       x=self.BN2(x)
    
       x=self.conv4(x)
       
       x=self.BN2_2(x)
    
       x=self.maxpool3(x)
       
    
       x=self.conv4_2(x)
       
       
       x=self.BN2_3(x)
    
       x=self.conv4_3(x)
       
       x=self.BN2_4(x)
       #x=tf.reshape(x,[tf.shape(x)[0],-1])
       x=keras.layers.Flatten()(x)
       #x=self.global_pool(x)
       #x=keras.layers.Dropout(0.5)(x)
       
       x=self.linear1(x)
       x=keras.layers.Dropout(0.5)(x)
       x=self.linear2(x)
      
       return x
    def homography(self,Xs,x):
 
         num_batch=tf.shape(Xs)[0]
         Xs=tf.cast(tf.reshape(Xs,[num_batch,8,1]),dtype=tf.float32)
         x=tf.cast(tf.reshape(x,[num_batch,8,1]),dtype=tf.float32)
         Xp=tf.add(Xs,x)
         #tf.keras.backend.print_tensor(Xp,'x is ')
         #tf.keras.backend.print_tensor(Xs,'xs is ')
         
         C_plus_x_1=[[0,0,0,0,0,0,0,0],
                      [1,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0],
                    [0,0,1,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,0]
                    ]
         C_plus_y_1=[[0,0,0,0,0,0,0,0],
                      [0,1,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0],
                    [0,0,0,1,0,0,0,0],
                    [0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,1]]
         C_plus_1=[0,1,0,1,0,1,0,1]
         C_plus_1=tf.reshape(C_plus_1,[-1,1])
         
         C_minus_x_1=[[-1,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0],
                    [0,0,-1,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0],
                    [0,0,0,0,-1,0,0,0],
                    [0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,-1,0],
                    [0,0,0,0,0,0,0,0]]
         
         C_minus_y_1=[[0,-1,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0],
                    [0,0,0,-1,0,0,0,0],
                    [0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,-1,0,0],
                    [0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,-1],
                    [0,0,0,0,0,0,0,0]]
         
         C_minus_1=[-1,0,-1,0,-1,0,-1,0]
         C_minus_1=tf.reshape(C_minus_1,[-1,1])
         C_plus_7=[[0,1,0,0,0,0,0,0],
                      [1,0,0,0,0,0,0,0],
                      [0,0,0,1,0,0,0,0],
                      [0,0,1,0,0,0,0,0],
                      [0,0,0,0,0,1,0,0],
                      [0,0,0,0,1,0,0,0],
                      [0,0,0,0,0,0,0,1],
                      [0,0,0,0,0,0,1,0]]
         

         C_b_7=[[1,0,0,0,0,0,0,0],
               [-1,0,0,0,0,0,0,0],
               [0,0,1,0,0,0,0,0],
               [0,0,-1,0,0,0,0,0],
               [0,0,0,0,1,0,0,0],
               [0,0,0,0,-1,0,0,0],
               [0,0,0,0,0,0,1,0],
               [0,0,0,0,0,0,-1,0]]
         C_b_8=[[0,1,0,0,0,0,0,0],
               [0,-1,0,0,0,0,0,0],
               [0,0,0,1,0,0,0,0],
               [0,0,0,-1,0,0,0,0],
               [0,0,0,0,0,1,0,0],
               [0,0,0,0,0,-1,0,0],
               [0,0,0,0,0,0,0,1],
               [0,0,0,0,0,0,0,-1]]
         C_out=[[0,-1,0,0,0,0,0,0],
                      [1,0,0,0,0,0,0,0],
                      [0,0,0,-1,0,0,0,0],
                      [0,0,1,0,0,0,0,0],
                      [0,0,0,0,0,-1,0,0],
                      [0,0,0,0,1,0,0,0],
                      [0,0,0,0,0,0,0,-1],
                      [0,0,0,0,0,0,1,0]]
         
         
         C_minus_x_1=tf.cast(C_minus_x_1,tf.float32)
         C_minus_y_1=tf.cast(C_minus_y_1,tf.float32)
         C_minus_1=tf.cast(C_minus_1,tf.float32)

         
         C_plus_1=tf.cast(C_plus_1,tf.float32)
         C_plus_x_1=tf.cast(C_plus_x_1,tf.float32)
         C_plus_y_1=tf.cast(C_plus_y_1,tf.float32)

         C_b_7=tf.cast(C_b_7,tf.float32)
         C_b_8=tf.cast(C_b_8,tf.float32)
         C_plus_7=tf.cast(C_plus_7,tf.float32)
         C_out=tf.cast(C_out,tf.float32)
         
         C_minus_x_1=add_batch_size(C_minus_x_1,num_batch)
         C_minus_y_1=add_batch_size(C_minus_y_1,num_batch)
         C_minus_1=add_batch_size(C_minus_1,num_batch)
         C_plus_x_1=add_batch_size(C_plus_x_1,num_batch)
         C_plus_y_1=add_batch_size(C_plus_y_1,num_batch)
         C_plus_1=add_batch_size(C_plus_1,num_batch)
         C_b_7=add_batch_size(C_b_7,num_batch)
         C_b_8=add_batch_size(C_b_8,num_batch)
         C_out=add_batch_size(C_out,num_batch)
         A1=tf.matmul(C_plus_x_1,Xs)

         A2=tf.matmul(C_plus_y_1,Xs)

         A3=C_plus_1

         A4=tf.matmul(C_minus_x_1,Xs)

         A5=tf.matmul(C_minus_y_1,Xs)

         A6=C_minus_1

         A7=tf.matmul(C_plus_7,Xs)*tf.matmul(C_b_7,Xp)

         A8=tf.matmul(C_plus_7,Xs)*tf.matmul(C_b_8,Xp)

         
         
         A=tf.concat([A1,A2,A3,A4,A5,A6,A7,A8],-1)
         
         rhs=tf.matmul(C_out,Xp)

         H=tf.linalg.solve(A, rhs)
         H9=tf.ones([num_batch,1,1],dtype=tf.float32)
         H=tf.concat([H,H9],1)
         H=tf.reshape(H,[num_batch,3,3])
         return H  
    # @tf.function
    # def train_step(self,input_,label):

    #     with tf.GradientTape() as tape:
    #         predictions = self(input_,training=True)
    #         loss=self.loss(label,predictions)

    #     print('1')
    #     gradients=tape.gradient(loss,model.trainable_variables)

    #     optimizer.apply_gradients(zip(gradients,model.trainable_variables))

                                
    def SparseCategoricalAccuracy(self):
        return tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    def sparse_categorical_loss(self,y_t,y_p):
        return tf.keras.losses.sparse_categorical_crossentropy(from_logits=True)
    def optimizer_(self):
        return keras.optimizers.Adam()
    def model_(self,modeltype='sup'):
      inputs1=keras.layers.Input(shape=(128,128,2,))
      inputs2=keras.layers.Input(shape=(128,128,1,))
      inputs=(inputs1,inputs2)
      if modeltype=='sup':
        return keras.Model(inputs1,self.supervised_model(inputs1))
      else:
        return keras.Model(inputs,self.unsupervised_model(inputs))  
    def _meshgrid(self,num_batch,height,width,scale_h):
        with tf.name_scope('meshgrid'):    
            height=tf.cast(height,tf.int32)
            width=tf.cast(width,tf.int32)
            if scale_h:
                x_t=tf.linspace(-1.0,1.0,width)
                y_t=tf.linspace(-1.0,1.0,height)
            else:
                x_t=tf.linspace(0.0,width,width)
                y_t=tf.linspace(0.0,height,height)
            x_t,y_t=tf.meshgrid(x_t,y_t)
            x_t=tf.reshape(x_t,[1,-1])
            y_t=tf.reshape(y_t,[1,-1])
            
            ones=tf.ones(x_t.shape,dtype=tf.float32)
            stack=tf.concat([x_t,y_t,ones],0)
            
            output=tf.tile(tf.expand_dims(stack,0),[num_batch,1,1])
            
            return output  
    def _transform(self,theta, input_dim, out_size, scale_h):
        
        with tf.name_scope('transform'):
            height=input_dim.shape[1]
            width=input_dim.shape[2]
            
            theta_t=theta
            theta_t=tf.cast(theta_t,tf.float32)
            
            out_h=out_size[0]
            out_w=out_size[1]
            num_batches=tf.shape(input_dim)[0]
            grid=self._meshgrid(num_batches,out_h,out_w,scale_h)
            
            T_g= tf.matmul(theta_t,grid)
            
            x_g=tf.slice(T_g,[0,0,0],[-1,1,-1])
            y_g=tf.slice(T_g,[0,1,0],[-1,1,-1])
            t_g=tf.slice(T_g,[0,2,0],[-1,1,-1])
            t_g=tf.reshape(t_g,[-1])
            
            zero=tf.constant(0,dtype=tf.float32)
            one=tf.constant(1,dtype=tf.float32)
            
            smaller=tf.constant(1e-6,dtype=tf.float32)
            smallest=1e-7*(1-tf.cast(tf.greater_equal(tf.abs(t_g),smaller),dtype=tf.float32))
          
            t_s=t_g+smallest
            
        
            x_t=tf.reshape(x_g,[-1])/t_s
            y_t=tf.reshape(y_g,[-1])/t_s
            
            
            
            out_im=self._interpolate(input_dim,x_t,y_t,out_size,scale_h)
            return out_im
            
    def _repeat(self,in_dim,out_dim):
        with tf.name_scope('repeat'):
            out_ones=tf.cast(tf.ones([1,out_dim]),tf.int32)
        
            
            
            return tf.reshape(tf.matmul(tf.reshape(in_dim,[-1,1]),out_ones),[-1])
    
    def _interpolate(self,im, x, y, out_size,scale_h):
        
        with tf.name_scope('interpolate'):
            out_w=out_size[1]
            out_h=out_size[0]
            num_batches=tf.shape(im)[0]
            
            
            im_w=im.shape[2]
            im_c=im.shape[3]
            im_h=im.shape[1]
            max_x=tf.cast(im_w-1,dtype=tf.int32)
            max_y=tf.cast(im_h-1,dtype=tf.int32)
            min_=tf.constant(0,dtype=tf.int32)
            
            if scale_h:
                x=(x+1.0)*tf.cast(im_w,tf.float32)/2.0
                y=(y+1.0)*tf.cast(im_h,tf.float32)/2.0
            x_0=tf.cast(tf.floor(x),dtype=tf.int32)
            y_0=tf.cast(tf.floor(y),dtype=tf.int32)
            x_1=x_0+1
        
            y_1=y_0+1
            x_v0=tf.clip_by_value(x_0,min_,max_x)
            y_v0=tf.clip_by_value(y_0,min_,max_y)
            x_v1=tf.clip_by_value(x_1,min_,max_x)
            y_v1=tf.clip_by_value(y_1,min_,max_y)
        
            
            dim2=im_w
            dim1=im_w*im_h
            base=self._repeat(tf.range(num_batches)*dim1,out_w*out_h)
            
            base_y0=base+y_v0*dim2
            base_y1=base+y_v1*dim2
            
            idx_a=base_y0+x_v0
            idx_b=base_y1+x_v0
            idx_c=base_y0+x_v1
            idx_d=base_y1+x_v1
        
            idx_a=tf.cast(idx_a,dtype=tf.int32)
            idx_b=tf.cast(idx_b,dtype=tf.int32)
            idx_c=tf.cast(idx_c,dtype=tf.int32)    
            idx_d=tf.cast(idx_d,dtype=tf.int32)
            
            im_flat=tf.cast(tf.reshape(im,[-1,1,]),dtype=tf.float32)
            Im_a=tf.gather(im_flat,idx_a)
            Im_b=tf.gather(im_flat,idx_b)
            Im_c=tf.gather(im_flat,idx_c)
            Im_d=tf.gather(im_flat,idx_d)
              
            x_v0=tf.cast(x_v0,dtype=tf.float32)
            y_v0=tf.cast(y_v0,dtype=tf.float32)
            x_v1=tf.cast(x_v1,dtype=tf.float32)    
            y_v1=tf.cast(y_v1,dtype=tf.float32)
            
            IA=tf.expand_dims((x_v1-x)*(y_v1-y),1)*Im_a
            IB=tf.expand_dims((x_v1-x)*(y-y_v0),1)*Im_b
            IC=tf.expand_dims((x-x_v0)*(y_v1-y),1)*Im_c
            ID=tf.expand_dims((x-x_v0)*(y-y_v0),1)*Im_d
            
            output=tf.add_n([IA,IB,IC,ID])
 
            output=tf.reshape(output,tf.stack([num_batches,out_h,out_w,1]),name='best')
        
        return output
    
    
    def batch_transform(self,input_im,thetas,out_size,name='BatchSpatialTransformer'):

        return self._transform(thetas,input_im,out_size,scale_h=True)

def add_batch_size(tensor,num_batch):
    return tf.tile(tf.expand_dims(tensor,0),tf.stack([num_batch,1,1]))