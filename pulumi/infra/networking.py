"""Default VPC networking for the scheduled ECS task."""

import pulumi_aws as aws

from . import config

default_vpc = aws.ec2.get_vpc(default=True)
default_vpc_subnets = aws.ec2.get_subnets_output(
    filters=[aws.ec2.GetSubnetsFilterArgs(name="vpc-id", values=[default_vpc.id])]
)
primary_subnet_id = default_vpc_subnets.ids.apply(lambda ids: ids[0] if ids else None)

ecs_sg = aws.ec2.SecurityGroup(
    "cdt-ecs-sg",
    name=f"{config.name_prefix}-sg-ecs",
    description="Security group for CDT ECS Fargate tasks",
    vpc_id=default_vpc.id,
    ingress=[],
    egress=[
        aws.ec2.SecurityGroupEgressArgs(
            from_port=0,
            to_port=0,
            protocol="-1",
            cidr_blocks=["0.0.0.0/0"],
        )
    ],
    tags=config.tags(),
)
