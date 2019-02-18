class MyMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        response["Access-Control-Allow-Origin"] = "localhost:3000/login"
        response["Access-Control-Allow-Methods"] = "POST"
        response[
            "Access-Control-Allow-Headers"
        ] = "Content-Type, Accept, X-CSRFToken, Authorization"

        return response
