from ultralytics import YOLO
import time
import os
import argparse
import torch
import tracemalloc
from multiprocessing import Pool

class Tester():
    def __init__(self, args):
        self.data_path = args.data_path
        self.model_base_path = args.model_path
        self.duration = args.duration
        self.threads = args.threads
        self.batch_size = args.batch_size
        self.num_models = args.num_models
        self.regions = ['10S', '10T', '11R', '12R', '16T', '17R', '17T', '18S', '32S', '32T',
        '33S', '33T', '52S', '53S', '54S', '54T']

        self.model = self.load_model(args.region)
    

    def load_model(self, region):
        # load model checkpoint trained on a specific region
        start = time.time()
        model = YOLO(os.path.join(self.model_base_path, region, region+'_nadir.pt'))
        end = time.time()
        print(f'Model checkpoint for region {region} loaded in {end-start} sec')
        return model
    

    def log(self, name, time, mem_results):
        # print results to terminal
        print(f'\n========== {name} results ==========')
        print(f'Run time: {time}')
        print(f'Current memory usage: {mem_results[0]}')
        print(f'Peak memory usage: {mem_results[1]}\n')
    

    def _single_run(self, model, gpu=False):
        # call YOLO model predict on a single image
        img_file = [file for file in os.listdir(self.data_path)][0]
        if gpu:
            results = model.predict(os.path.join(self.data_path, img_file), device='cuda:0')
        else:
            results = model.predict(os.path.join(self.data_path, img_file))
        
        return results


    def test_single(self):
        # test single model on a single image
        tracemalloc.start()

        start = time.time()
        results = self._single_run(self.model)
        end = time.time()
        
        # log time and memory
        mem_results = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.log('Test single run', end-start, mem_results)

    
    def test_single_gpu(self):
        # test single model on a single image on GPU
        tracemalloc.start()
        
        start = time.time()
        results = self._single_run(self.model, gpu=True)
        end = time.time()
        
        # log time and memory
        mem_results = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.log('Test single run GPU', end-start, mem_results)


    def test_batch(self):
        # test a batch of images at once with a single model
        tracemalloc.start()
        images = [os.path.join(self.data_path, file) for file in os.listdir(self.data_path)][:self.batch_size]
        
        start = time.time()
        results = self.model.predict(images)
        end = time.time()

        # log time and memory
        mem_results = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.log('Test batch', end-start, mem_results)


    def test_duration(self):
        # run a single model on a single image for a specified duration of time
        tracemalloc.start()

        start = time.time()
        end = time.time()

        while (end-start) < self.duration:
            results = self._single_run(self.model)
            end = time.time()
        
        # log time and memory
        mem_results = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.log('Test duration', end-start, mem_results)


    def test_multiple_sequential(self):
        # load and run multiple models sequentially
        tracemalloc.start()

        start = time.time()
        regions = self.regions[:self.num_models]
        for r in regions:
            model = self.load_model(r)
            results = self._single_run(model)

        end = time.time()
        
        # log time and memory
        mem_results = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.log('Test multiple sequential', end-start, mem_results)
        

    def test_multiple_parallel(self):
        # load and run multiple models in parallel
        tracemalloc.start()

        start = time.time()
        regions = self.regions[:self.num_models]
        region_models = [self.load_model(r) for r in regions]

        with Pool(self.num_models) as p:
            results = p.map(self._single_run, region_models)

        end = time.time()
        
        # log time and memory
        mem_results = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.log('Test multiple parallel', end-start, mem_results)


    def test_parallel(self):
        # run a single model on a single image in parallel using threads
        tracemalloc.start()

        start = time.time()
        with Pool(self.threads) as p:
            args = [self.model for i in range(self.threads)]
            results = p.map(self._single_run, args)
        end = time.time()
        
        # log time and memory
        mem_results = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.log('Test parallel', end-start, mem_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test dataloader functions")
    parser.add_argument("--data_path", type=str, required=True, help='path to folder of images, eg. 17R_s2_val')
    parser.add_argument('--model_path', type=str, required=True, help='path to model checkpoints for each region, eg. NN-models-main/ld')
    parser.add_argument('--region', type=str, default='17R', help='region key for model checkpoint')
    parser.add_argument('--duration', type=int, default=10, help='duration of time in seconds for duration test')
    parser.add_argument('--threads', type=int, default=5, help='number of threads to use in parallel tests')
    parser.add_argument('--batch_size', type=int, default=8, help='number of images to use in a batch for batch test')
    parser.add_argument('--num_models', type=int, default=5, help='number of models to load for multiple model tests')
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    tester = Tester(args)

    # run all tests
    tester.test_parallel()
    tester.test_multiple_parallel()
    tester.test_multiple_sequential()
    tester.test_single()
    tester.test_single_gpu()
    tester.test_duration()
    tester.test_batch()