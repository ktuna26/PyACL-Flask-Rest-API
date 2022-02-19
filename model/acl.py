"""
This script provide asynchronous inference, it collects 
a couple of image from queue and process them

Copyright 2022 Huawei Technologies Co., Ltd

CREATED:  2022-02-06 20:12:13
MODIFIED: 2022-02-09 22:30:45
"""
# -*- coding:utf-8 -*-
import acl

from utils.acl_util import check_ret
from data.constant import ACL_MEM_MALLOC_NORMAL_ONLY, \
    ACL_MEMCPY_HOST_TO_DEVICE, ACL_MEMCPY_DEVICE_TO_HOST
from utils.postprocessing import detect, non_max_suppression, \
                                scale_coords


class NET(object):
    
    def __init__(self, 
                device_id,
                model_path):
        self.device_id = device_id    # int
        self.model_path = model_path  # string

        self.input_size = 0
        self.output_size = 0
        self.class_num = 0
        self.img_dims = None
        
        self.model_id = None    # pointer
        self.context = None     # pointer
        self.stream = None      # pointer
        self.model_desc = None  # pointer when using
        self.input_data = None
        self.output_data = None
       
        self.model_output_dims = []
        self.dataset_list = []

        self.exit_flag = False

        self.__init_resource()
        
        
    def __del__(self):
        print('[ACL] release source stage:')
        if self.model_id:
            ret = acl.mdl.unload(self.model_id)
            check_ret("acl.mdl.unload", ret)

        if self.model_desc:
            ret = acl.mdl.destroy_desc(self.model_desc)
            check_ret("acl.mdl.destroy_desc", ret)

        self.__destroy_dataset_and_databuf()

        if self.stream:
            result = acl.rt.destroy_stream(self.stream)
            check_ret("acl.rt.destroy_stream", result)

        if self.context:
            result = acl.rt.destroy_context(self.context)
            check_ret("acl.rt.destroy_context", result)

        ret = acl.rt.reset_device(self.device_id)
        check_ret("acl.rt.reset_device", ret)
        ret = acl.finalize()
        check_ret("acl.finalize", ret)
        print('[ACL] release source stage success')
        
        
    def __init_resource(self):
        print("[ACL] init resource stage:")
        ret = acl.init()
        check_ret("acl.init", ret)
            
        ret = acl.rt.set_device(self.device_id)
        check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        check_ret("acl.rt.create_context", ret)

        self.stream, ret = acl.rt.create_stream()
        check_ret("acl.rt.create_stream", ret)
        print("[ACL] init resource stage success")
        
        self.__get_model_info()
      
    
    def __get_model_info(self,):
        print("[MODEL] model init resource stage:")
        self.model_id, ret = acl.mdl.load_from_file(self.model_path) # load model
        check_ret("acl.mdl.load_from_file", ret)
        
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)
        
        self.input_size = acl.mdl.get_num_inputs(self.model_desc)
        self.output_size = acl.mdl.get_num_outputs(self.model_desc)
        print("=" * 90)
        
        print("[MODEL] model input size", self.input_size)
        for i in range(self.input_size):
            print(">> input ", i)
            print("model input dims", acl.mdl.get_input_dims(self.model_desc, i))
            print("model input datatype", acl.mdl.get_input_data_type(self.model_desc, i))
            self.model_input_height, self.model_input_width = (i * 2 for i in acl.mdl.get_input_dims(self.model_desc, i)[0]['dims'][2:])
        print("=" * 90)
        
        print("[MODEL] model output size", self.output_size)
        for i in range(self.output_size):
            print(">> output ", i)
            print("model output dims", acl.mdl.get_output_dims(self.model_desc, i))
            print("model output datatype", acl.mdl.get_output_data_type(self.model_desc, i))
            self.class_num = acl.mdl.get_output_dims(self.model_desc, i)[0]['dims'][2]
            self.model_output_dims.append(tuple(acl.mdl.get_output_dims(self.model_desc, i)[0]['dims']))
        print("=" * 90)

        print("[MODEL] class Model init resource stage success")
        print("=" * 90)
        
        
    def __load_input_data(self, images_data):
        print("[MODEL] create model input dataset:")
        img_ptr = acl.util.numpy_to_ptr(images_data)  # host ptr
        image_buffer_size = images_data.size * images_data.itemsize
        print("[ACL] img_host_ptr, img_buf_size: ", img_ptr, image_buffer_size)
        # memcopy host to device
        img_device, ret = acl.rt.malloc(image_buffer_size, 
                                        ACL_MEM_MALLOC_NORMAL_ONLY)
        check_ret("acl.rt.malloc", ret)
        
        ret = acl.rt.memcpy(img_device, image_buffer_size, img_ptr,
                            image_buffer_size, ACL_MEMCPY_HOST_TO_DEVICE)
        check_ret("acl.rt.memcpy", ret)
        print("[MODEL] create model input dataset success")

        # create dataset in device
        print("[MODEL] create model input dataset:")
        img_dataset = acl.mdl.create_dataset()
        img_data_buffer = acl.create_data_buffer(img_device, image_buffer_size)
        
        if img_data_buffer is None:
            print("[ERROR] can't create data buffer, create input failed!!!")

        _, ret = acl.mdl.add_dataset_buffer(img_dataset, 
                                            img_data_buffer)
        if ret:
            ret = acl.destroy_data_buffer(img_data_buffer)
            check_ret("acl.destroy_data_buffer", ret)
        print("[MODEL] create model input dataset success")
        
        return img_dataset
    
    
    def __load_output_data(self):
        print("[MODEL] create model output dataset:")
        output_data = acl.mdl.create_dataset()
        for i in range(self.output_size):
            # check temp_buffer dtype
            temp_buffer_size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            temp_buffer, ret = acl.rt.malloc(temp_buffer_size, 
                                            ACL_MEM_MALLOC_NORMAL_ONLY)
            check_ret("acl.rt.malloc", ret)
            
            data_buf = acl.create_data_buffer(temp_buffer, 
                                            temp_buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(output_data, data_buf)
            
            if ret:
                ret = acl.destroy_data_buffer(data_buf)
                check_ret("acl.destroy_data_buffer", ret)
        print("[MODEL] create model output dataset success")
        return output_data
    
    
    def __data_interaction(self, images_dataset_list):
        print("[ACL] data interaction from host to device")
        for image_data in images_dataset_list:
            img_input = self.__load_input_data(image_data)
            infer_ouput = self.__load_output_data()
            self.dataset_list.append([img_input, infer_ouput])
        print("[ACL] data interaction from host to device success")
        
        
    def __destroy_dataset_and_databuf(self, ):
        while self.dataset_list:
            dataset = self.dataset_list.pop()
            for temp in dataset:
                num_temp = acl.mdl.get_dataset_num_buffers(temp)
                for i in range(num_temp):
                    data_buf_temp = acl.mdl.get_dataset_buffer(temp, i)
                    if data_buf_temp:
                        data = acl.get_data_buffer_addr(data_buf_temp)
                        ret = acl.rt.free(data)
                        check_ret("acl.rt.free", ret)
                        ret = acl.destroy_data_buffer(data_buf_temp)
                        check_ret("acl.destroy_data_buffer", ret)
                ret = acl.mdl.destroy_dataset(temp)
                check_ret("acl.mdl.destroy_dataset", ret)
                
                
    def __process_callback(self, args_list):
        context, time_out = args_list

        acl.rt.set_context(context)
        while True:
            acl.rt.process_report(time_out)
            if self.exit_flag:
                print("[ACL] exit acl.rt.process_report")
                break


    def __get_callback(self):
        acl.rt.launch_callback(self.__callback_func,
                            self.dataset_list,
                            1,
                            self.stream)
        
        
    def __forward(self):
        print('[MODEL] execute stage:')
        for img_data, infer_output in self.dataset_list:
            ret = acl.mdl.execute_async(self.model_id,
                                        img_data,
                                        infer_output,
                                        self.stream)
            check_ret("acl.mdl.execute_async", ret)

        self.__get_callback()
        print('[MODEL] execute stage success')
        
        
    def __callback_func(self, delete_list):
        print('[MODEL] callback func stage:')
        j = 0
        self.results = []
        for temp in delete_list:
            _, infer_output = temp
            
            # device to host
            feature_maps = []
            num = acl.mdl.get_dataset_num_buffers(infer_output)
            for i in range(num):
                temp_output_buf = acl.mdl.get_dataset_buffer(infer_output, i)

                infer_output_ptr = acl.get_data_buffer_addr(temp_output_buf)
                infer_output_size = acl.get_data_buffer_size_v2(temp_output_buf)

                output_host, ret = acl.rt.malloc_host(infer_output_size)
                check_ret("acl.rt.malloc_host", ret)
                
                ret = acl.rt.memcpy(output_host,
                                    infer_output_size,
                                    infer_output_ptr,
                                    infer_output_size,
                                    ACL_MEMCPY_DEVICE_TO_HOST)
                check_ret("acl.rt.memcpy", ret)
                
                feature_maps.append((acl.util.ptr_to_numpy(output_host, (infer_output_size//4,), 
                                    11).reshape(self.model_output_dims[i])).transpose((0, 1, 3, 4, 2)))
                
            res_tensor = detect(feature_maps, self.class_num)
            
            # Apply NMS
            pred = non_max_suppression(res_tensor, conf_thres=0.33, iou_thres=0.5)
            
            # Process detections
            bboxes = []
            for i, det in enumerate(pred):  # detections per image
                # Rescale boxes from img_size to im0 size
                if det is not None:
                    det[:, :4] = scale_coords((self.model_input_width, self.model_input_height), 
                                            det[:, :4], self.img_dims[j]).round()
                    for *xyxy, conf, cls in det:
                        bboxes.append([*xyxy, conf, int(cls)])
                else:
                    pass
            j += 1

            self.results.append(bboxes)

        self.__destroy_dataset_and_databuf()
        print('[MODEL] callback func stage success')

    
    def run(self, img_datas, img_dims):
        if not isinstance(img_datas, list):
            print("[ERROR] images isn't list")
            return None

        self.img_dims = img_dims
        # copy images to device
        self.__data_interaction(img_datas)

        # thread
        tid, ret = acl.util.start_thread(self.__process_callback,
                                         [self.context, 50])
        check_ret("acl.util.start_thread", ret)

        ret = acl.rt.subscribe_report(tid, self.stream)
        check_ret("acl.rt.subscribe_report", ret)

        self.__forward()
        ret = acl.rt.synchronize_stream(self.stream)
        check_ret("acl.rt.synchronize_stream", ret)
        ret = acl.rt.unsubscribe_report(tid, self.stream)
        check_ret("acl.rt.unsubscribe_report", ret)
        
        self.exit_flag = True
        ret = acl.util.stop_thread(tid)
        check_ret("acl.util.stop_thread", ret)
        
        return self.results

    
    def get_model_input_dims(self):
        return self.model_input_width, self.model_input_height