from grpc_interceptor import ServerInterceptor


# TODO: Implement!
class ErrorLogger(ServerInterceptor):
    def intercept(self, method, request, context, method_name):
        try:
            return method(request, context)
        except Exception as e:
            self.log_error(e)
            raise

    #def log_error(self, e: Exception) -> None: