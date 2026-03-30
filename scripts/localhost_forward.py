import argparse
import socket
import threading


def pipe(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except Exception:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except Exception:
            pass


def handle_client(client: socket.socket, target_host: str, target_port: int) -> None:
    upstream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        upstream.connect((target_host, target_port))
    except Exception:
        client.close()
        upstream.close()
        return

    t1 = threading.Thread(target=pipe, args=(client, upstream), daemon=True)
    t2 = threading.Thread(target=pipe, args=(upstream, client), daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    client.close()
    upstream.close()


def serve(listen_host: str, listen_port: int, target_host: str, target_port: int) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((listen_host, listen_port))
    server.listen(128)
    print(f"Forwarding {listen_host}:{listen_port} -> {target_host}:{target_port}")
    while True:
        client, _ = server.accept()
        threading.Thread(target=handle_client, args=(client, target_host, target_port), daemon=True).start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-host", required=True)
    parser.add_argument("--ports", nargs="+", type=int, required=True)
    parser.add_argument("--listen-host", default="127.0.0.1")
    args = parser.parse_args()

    for p in args.ports:
        threading.Thread(target=serve, args=(args.listen_host, p, args.target_host, p), daemon=True).start()

    threading.Event().wait()
