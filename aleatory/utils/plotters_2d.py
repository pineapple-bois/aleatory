from matplotlib import pyplot as plt


# TODO: refactor `plot_paths_coordinates'to return axes for downstream
#  or deprecate and remove
def plot_paths_coordinates(
    *args,
    times,
    paths1,
    paths2,
    style="seaborn-v0_8-whitegrid",
    title=None,
    suptitle=None,
    mode="steps",
    **fig_kw,
):
    with plt.style.context(style):
        fig, ax = plt.subplots(**fig_kw)
        for p1, p2 in zip(paths1, paths2):
            if mode == "points":
                ax.scatter(times, p1, s=7)
                ax.scatter(times, p2, s=7)
            elif mode == "steps":
                ax.step(times, p1, where="post", label="$X_1$")
                ax.step(times, p2, where="post", label="$X_2$")
            elif mode == "linear":
                ax.plot(times, p1, label="$X_1$", *args)
                ax.plot(times, p2, label="$X_2$", *args)
            elif mode in ["points+steps", "steps+points"]:
                ax.step(times, p1, where="post")
                ax.step(times, p2, where="post")
                color = plt.gca().lines[-1].get_color()
                ax.plot(times, p1, "o", color=color)
                ax.plot(times, p2, "o", color=color)
            else:
                raise ValueError("mode must be 'points', 'steps', or 'points+steps'.")
        if suptitle is not None:
            fig.suptitle(suptitle)
        ax.set_title(title)
        ax.set_xlabel("$t$")
        ax.set_ylabel("Coordinate processes")
        ax.legend(loc="best")
        plt.show()
    return fig


def plot_coordinate_paths(
    *args,
    times,
    paths1,
    paths2,
    style="seaborn-v0_8-whitegrid",
    title=None,
    suptitle=None,
    mode="steps",
    ax=None,
    **fig_kw,
):
    with plt.style.context(style):
        created_fig = False

        if ax is None:
            fig, ax = plt.subplots(**fig_kw)
            created_fig = True
        else:
            fig = ax.figure

        for i, (p1, p2) in enumerate(zip(paths1, paths2)):
            label1 = "$X_1$" if i == 0 else None
            label2 = "$X_2$" if i == 0 else None

            if mode == "points":
                ax.scatter(times, p1, s=7, label=label1)
                ax.scatter(times, p2, s=7, label=label2)
            elif mode == "steps":
                ax.step(times, p1, where="post", label=label1)
                ax.step(times, p2, where="post", label=label2)
            elif mode == "linear":
                ax.plot(times, p1, *args, label=label1)
                ax.plot(times, p2, *args, label=label2)
            elif mode in ["points+steps", "steps+points"]:
                ax.step(times, p1, where="post", label=label1)
                ax.step(times, p2, where="post", label=label2)
                color1 = ax.lines[-2].get_color()
                color2 = ax.lines[-1].get_color()
                ax.plot(times, p1, "o", color=color1)
                ax.plot(times, p2, "o", color=color2)
            else:
                raise ValueError(
                    "mode must be 'points', 'steps', 'linear', or 'points+steps'"
                )

        if created_fig and suptitle is not None:
            fig.suptitle(suptitle)

        ax.set_title(title)
        ax.set_xlabel("$t$")
        ax.set_ylabel("Coordinate processes")
        ax.legend(loc="best")

    return ax


def plot_sample_2d(
    path,
    style="seaborn-v0_8-whitegrid",
    title=None,
    suptitle=None,
    **fig_kw,
):
    x_positions, y_positions = path

    with plt.style.context(style):
        fig, ax = plt.subplots(**fig_kw)
        ax.plot(x_positions, y_positions, linewidth=1)
        ax.plot(x_positions[0], y_positions[0], "go", label="Start")  # Start point
        ax.plot(x_positions[-1], y_positions[-1], "ro", label="End")  # End point
        if suptitle is not None:
            fig.suptitle(suptitle)
        ax.set_title(title)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        plt.grid(True)
        plt.legend()
        plt.axis("equal")  # Equal scaling for both axes
        plt.show()

    return fig
