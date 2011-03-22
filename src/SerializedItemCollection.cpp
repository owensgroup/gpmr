#include <gpmr/SerializedItemCollection.h>
#include <algorithm>
#include <cstdio>
#include <cstring>

namespace gpmr
{
  const int SerializedItemCollection::INITIAL_CAPACITY = 1024 * 1024;
  SerializedItemCollection::SerializedItemCollection()
  {
    capacity = INITIAL_CAPACITY;
    totalSize = sizeof(int);
    storage = new char[capacity];
    numItems = reinterpret_cast<int * >(storage);
    *numItems = 0;
    lastItemIndex = -1;
  }
  SerializedItemCollection::SerializedItemCollection(const int initialCapacity)
  {
    capacity = std::max(INITIAL_CAPACITY, initialCapacity);
    totalSize = sizeof(int);
    storage = new char[capacity];
    numItems = reinterpret_cast<int * >(storage);
    *numItems = 0;
    lastItemIndex = -1;
  }
  SerializedItemCollection::SerializedItemCollection(const int pCapacity, const int pTotalSize, void * const relinquishedBuffer)
    : capacity(pCapacity),
      totalSize(pTotalSize),
      numItems(reinterpret_cast<int * >(relinquishedBuffer)),
      storage(reinterpret_cast<char * >(relinquishedBuffer))
  {
    lastItemIndex = -1;
  }
  SerializedItemCollection::~SerializedItemCollection()
  {
    delete [] storage;
  }
  // TBI
  /*
  void SerializedItemCollection::clear()


  int SerializedItemCollection::getTotalSize() const

  int SerializedItemCollection::getItemCount() const

  const void * SerializedItemCollection::getStorage() const

  void * SerializedItemCollection::getItem(const int itemNumber, int & key)

  const Ray * SerializedItemCollection::getItem(const int itemNumber, int & key) const


  void SerializedItemCollection::addItem(const int key, const Ray & value)

  void SerializedItemCollection::addItems(const int pTotalSize, const void * const buffer)
  */
}
