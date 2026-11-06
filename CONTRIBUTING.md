# Contributing

We are not accepting external pull requests yet. Bug reports and questions via Issues are
welcome. To report a security vulnerability, see [SECURITY.md](SECURITY.md).

The rest of this file is for maintainers.

## Build and test

GitHub CI only lints (ruff and shellcheck): the image takes hours to build on a hosted runner. Before you open a pull
request, run the build and the tests locally (README, Tests):

```sh
./scripts/oap test   # colcon build -Werror, pytest, the oap_control gtests, in a throwaway container
```

A change that only runs on the robot (`hardware:=real`, the gripper, the cameras) cannot be tested in the container:
say in the pull request how you tested it on the robot.

## Commits

- Use [Conventional Commits](https://www.conventionalcommits.org/): `fix:`, `feat:`, `docs:`,
  `chore:`, with an imperative subject.
- Use body bullets that explain why the change is needed.
- Never commit API keys, passwords, tokens, checkpoints or recorded bags.

## Licence

By contributing, you agree that your contribution is licensed under Apache-2.0, the licence of
this repository ([`LICENSE`](LICENSE)).
