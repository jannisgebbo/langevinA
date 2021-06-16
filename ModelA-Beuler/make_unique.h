#ifndef MODELA_MAKE_UNIQUE_H
#define MODELA_MAKE_UNIQUE_H

#define MODELA_NO_CPP14

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
#ifdef MODELA_NO_CPP14
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
#else 
  return std::make_unique<T>(std::forward<Args>(args)...);
#endif
}
#endif


