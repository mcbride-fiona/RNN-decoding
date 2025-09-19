def main():
    try:
        from .train import train
    except Exception as e:
        print(f"Failed to import train(): {e}")
        return 1
    try:
        train()
        return 0
    except Exception as e:
        print(f"Error during training: {e}")
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
