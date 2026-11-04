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
            product = graph.add(f"{f}*{other}", "power", [roots[f], roots[other]])
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


def grow_2d(graph):
    fields = graph.fields
    axes = (("x", "Dx"), ("y", "Dy"))
    roots = {field: graph.add(field, "power", layer="L0") for field in fields}
    for field in fields:
        for power in range(1, 6):
            base = graph.add(f"{field}**{power}", "power", [roots[field]])
            if power <= 3:
                for coordinate, op in axes:
                    for order in range(1, 5):
                        graph.add(f"{op}({field}**{power},{order})", "differentiate", [base])
                for nx in range(1, 4):
                    for ny in range(1, 5 - nx):
                        graph.add(
                            f"Dx(Dy({field}**{power},{ny}),{nx})", "mixed_differentiate", [base]
                        )
                graph.add(f"Lap({field}**{power})", "differentiate", [base])
        for fun in ("sin", "cos", "tanh", "exp"):
            graph.add(f"np.{fun}(a*{field})", "local_transform", [roots[field]])
        for denominator in (f"a+{field}", f"a+{field}**2"):
            graph.add(f"{field}/({denominator})", "local_transform", [roots[field]])
        for other in fields:
            graph.add(f"{field}*{other}", "power", [roots[field], roots[other]])
            for coordinate, op in axes:
                gradient = graph.add(f"{op}({field},1)", "differentiate", [roots[field]])
                for denominator in (other, f"{other}**2", f"a+{other}", f"1+a*{other}**2"):
                    quotient = graph.add(
                        f"{op}({field},1)/({denominator})", "quotient", [gradient, roots[other]]
                    )
                    graph.add(f"{op}({op}({field},1)/({denominator}),1)", "quotient", [quotient])
                graph.add(
                    f"{op}({field}*{op}({other},1),1)", "product_flux", [roots[field], roots[other]]
                )
    for field in ["1"] + fields:
        for power in [1] if field == "1" else [1, 2, 3]:
            base = f"({field})**{power}"
            for coordinate, op in axes:
                for modulation in (
                    f"np.sin(k*{coordinate})",
                    f"np.cos(k*{coordinate})",
                    f"np.tanh(k*{coordinate})",
                    f"np.exp(-k*{coordinate}**2)",
                    f"np.exp(k*{coordinate})",
                ):
                    node = graph.add(f"({base})*({modulation})", "modulate")
                    if field != "1":
                        graph.add(f"{op}(({base})*({modulation}),1)", "modulate", [node])
    for field in fields:
        for power in range(1, 4):
            response = f"{field}*(1-a*{field}**{power})"
            node = graph.add(response, "power", [roots[field]])
            for coordinate, op in axes:
                graph.add(f"{op}({response},1)", "differentiate", [node])
                for modulation in (
                    f"np.sin(k*{coordinate})",
                    f"np.cos(k*{coordinate})",
                    f"np.tanh(k*{coordinate})",
                    f"np.exp(-k*{coordinate}**2)",
                    f"np.exp(k*{coordinate})",
                ):
                    for factor in (modulation, f"(1-b*({modulation}))"):
                        modulated = graph.add(f"({response})*({factor})", "modulate", [node])
                        graph.add(f"{op}(({response})*({factor}),1)", "modulate", [modulated])
