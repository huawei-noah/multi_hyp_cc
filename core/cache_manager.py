# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved. THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import sys

# Class to cache info
class Cache():
    def __init__(self, length=float('inf')):
        self.length = length
        self.cache = {}

    def is_cached(self, key):
        return key in self.cache

    def read(self, key):
        return self.cache[key]

    def reset(self):
        return self.cache.clear()

    def save(self, key, value):
        if len(self.cache) < self.length:
            self.cache[key] = value

class CacheManager:
    def __init__(self, conf, no_cache = False):
        # Ideally, we want to cache all information of a dataset AFTER applying
        # transformations (resize, blacklevel subtraction, etc...). This speeds up
        # execution. However, this should not be done if your transformations
        # are random (RandomCrop, etc), because then, they won't be random.
        # That is why we check that 'transforms_valtest' is None, typically,
        # it will be != None when we have random transformations that we won't
        # be applying for validation and test.
        cache_transforms = conf['cache_transforms']
        if cache_transforms and conf['transforms_valtest'] is not None:
            print("ERROR: if 'cache_transforms' is enabled, 'transforms_valtest' should be null (forced equal to 'transforms')")
            sys.exit(-1)

        # if we don't want cache, simply set the cache size to 0
        if no_cache:
            self._cache_transforms = Cache(0)
            self._cache_dataset = Cache(0)
        else:
            # there are 2 caches: "cache_transforms"=after image transformations
            # and "cache_dataset"=images as read from disk (output of dataset class)
            # If we enable one, we disable the other.
            if cache_transforms:
                self._cache_transforms = Cache()
                self._cache_dataset = Cache(0)
            else:
                self._cache_transforms = Cache(0)
                self._cache_dataset = Cache()

    def reset(self):
        self._cache_transforms.reset()
        self._cache_dataset.reset()

    def transforms(self):
        return self._cache_transforms

    def dataset(self):
        return self._cache_dataset
