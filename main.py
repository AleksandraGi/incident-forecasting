from src.train import run_training_pipeline


def main():
    print("Starting incident forecasting pipeline...\n")

    run_training_pipeline(
        window_size=12,
        horizon=6,
        test_size=0.2,
        thresholds=[0.1, 0.3, 0.5, 0.7, 0.9],
    )

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()