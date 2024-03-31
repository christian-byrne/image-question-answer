from typing import Callable, Type, Union, Tuple

def raises_exception(exception_type: Union[Type[BaseException], Tuple[Type[BaseException], ...]]) -> Callable:
    def decorator(func: Callable) -> Callable:
        func.__annotations__["Raises"] = exception_type
        return func
    return decorator

