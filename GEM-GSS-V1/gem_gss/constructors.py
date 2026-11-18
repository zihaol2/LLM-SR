"""Original fixed 1D and 2D constructor rules; no reference inputs."""


def grow_1d(graph):
    """Uniform small constructor closure, independent of task identity and truth."""
    fields = graph.fields
    roots = {f: graph.add(f, "power", layer="L0") for f in fields}
    for f in fields:
        for p in range(1, 6):
            base = graph.add(f"{f}**{p}", "power", [roots[f]])
            if p <= 3:
                for order in range(1, 5):
                    graph.add(f"D({f}**{p},{order})", "differentiate", [base])
        for fun in ("sin", "cos", "tanh", "exp"):
            graph.add(f"np.{fun}(a*{f})", "local_transform", [roots[f]])
        for denominator in (f"a+{f}", f"a+{f}**2"):
            graph.add(f"{f}/({denominator})", "local_transform", [roots[f]])
        for other in fields:
            graph.add(f"{f}*{other}", "power", [roots[f], roots[other]])
            gradient = graph.add(f"D({f},1)", "differentiate", [roots[f]])
            for denominator in (other, f"{other}**2", f"a+{other}", f"1+a*{other}**2"):
                div = graph.add(f"D({f},1)/({denominator})", "quotient", [gradient, roots[other]])
                graph.add(f"D(D({f},1)/({denominator}),1)", "quotient", [div])
            graph.add(f"D({f}*D({other},1),1)", "product_flux", [roots[f], roots[other]])
    for f in ["1"] + fields:
        powers = [1] if f == "1" else [1, 2, 3]
        for p in powers:
            base = f"({f})**{p}"
            for modulation in [
                "np.sin(k*x)",
                "np.cos(k*x)",
                "np.tanh(k*x)",
                "np.exp(-k*x**2)",
                "np.exp(k*x)",
            ]:
                mid = graph.add(f"({base})*({modulation})", "modulate")
                if f != "1":
                    graph.add(f"D(({base})*({modulation}),1)", "modulate", [mid])
    for f in fields:
        for p in range(1, 4):
            response = f"{f}*(1-a*{f}**{p})"
            rid = graph.add(response, "power", [roots[f]])
            graph.add(f"D({response},1)", "differentiate", [rid])
            for modulation in [
                "np.sin(k*x)",
                "np.cos(k*x)",
                "np.tanh(k*x)",
                "np.exp(-k*x**2)",
                "np.exp(k*x)",
            ]:
                for factor in [modulation, f"(1-b*({modulation}))"]:
                    mid = graph.add(f"({response})*({factor})", "modulate", [rid])
                    graph.add(f"D(({response})*({factor}),1)", "modulate", [mid])


def grow_spatial(graph):
    """Finite coordinate-independent constructor for 2D and 3D scalar components."""
    from .grammar import multi_indices

    fields, coordinates = graph.fields, graph.coordinates
    dimensions = len(coordinates)
    roots = {field: graph.add(field, "power") for field in fields}

    def partial(text, axis, order=1):
        alpha = tuple(order if j == axis else 0 for j in range(dimensions))
        return "PD(" + text + "," + repr(alpha) + ")"

    for field in fields:
        for power in range(1, 6):
            text = f"{field}**{power}"
            base = graph.add(text, "power", [roots[field]])
            if power <= 3:
                for alpha in multi_indices(dimensions, 4):
                    family = (
                        "mixed_differentiate" if sum(n > 0 for n in alpha) > 1 else "differentiate"
                    )
                    graph.add(f"PD({text},{alpha})", family, [base])
                graph.add(f"Lap({text})", "differentiate", [base])
        for function in ("Sin", "Cos", "Tanh", "Exp"):
            graph.add(f"{function}(a*{field})", "local_transform", [roots[field]])
        for denominator in (f"a+{field}", f"a+{field}**2"):
            graph.add(f"{field}/({denominator})", "local_transform", [roots[field]])
        for other in fields:
            graph.add(f"{field}*{other}", "power", [roots[field], roots[other]])
            for axis in range(dimensions):
                gradient = partial(field, axis)
                for denominator in (
                    other,
                    f"{other}**2",
                    f"a+{other}",
                    f"1+a*{other}**2",
                ):
                    quotient = f"{gradient}/({denominator})"
                    node = graph.add(quotient, "quotient")
                    graph.add(partial(quotient, axis), "quotient", [node])
                graph.add(partial(f"{field}*{partial(other, axis)}", axis), "product_flux")

    def modulations(coordinate):
        return (
            f"Sin(k*{coordinate})",
            f"Cos(k*{coordinate})",
            f"Tanh(k*{coordinate})",
            f"Exp(-k*{coordinate}**2)",
            f"Exp(k*{coordinate})",
        )

    for field in ["1"] + fields:
        for power in [1] if field == "1" else [1, 2, 3]:
            for axis, coordinate in enumerate(coordinates):
                for modulation in modulations(coordinate):
                    text = f"({field}**{power})*({modulation})"
                    node = graph.add(text, "modulate")
                    if field != "1":
                        graph.add(partial(text, axis), "modulate", [node])
    for field in fields:
        for power in range(1, 4):
            response = f"{field}*(1-a*{field}**{power})"
            node = graph.add(response, "power", [roots[field]])
            for axis, coordinate in enumerate(coordinates):
                graph.add(partial(response, axis), "differentiate", [node])
                for modulation in modulations(coordinate):
                    for factor in (modulation, f"1-b*({modulation})"):
                        text = f"({response})*({factor})"
                        modulated = graph.add(text, "modulate", [node])
                        graph.add(partial(text, axis), "modulate", [modulated])
