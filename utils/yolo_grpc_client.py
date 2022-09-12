import grpc
import yolov5_service_pb2 as yolov5_service
import yolov5_service_pb2_grpc as yolov5_service_grpc


class YoloClient():
    def __init__(self, target):
        # Grpc service address
        self.target = target

    def call_detection_service(self, img):

        with grpc.insecure_channel(self.target) as channel:
            stub = yolov5_service_grpc.YoloV5Stub(channel)
            try:
                return self.detect_objects(stub, img)

            except grpc.RpcError as rpc_error:
                raise ValueError(rpc_error.code(), rpc_error.details())

    def detect_objects(self, stub, image_path):
        """
        Detects the objects in a single image
        by calling the server

        Returns:
        The DetectedPoses proto message with the
        estimated poses

        """
        print(f'Estimating image: \'{image_path}\'')
        with open(image_path, 'rb') as fp:
            image_bytes = fp.read()
            request = yolov5_service.Image(data=image_bytes)
        return stub.detect(request)
